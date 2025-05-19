import os
import json
import time
import datetime
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torchvision import transforms

from clip import clip
from timm.models.vision_transformer import vit_base_patch16_224, vit_base_patch16_384, vit_large_patch16_224

from datasets import DATASET_GETTERS
from models import *

from utils.meter import AverageMeter
from utils.samplers import DownSampler
from utils.losses import *
from utils.evaluator import Evaluator
from utils.templates import ZEROSHOT_TEMPLATES
from utils.misc import *


def load_clip_to_cpu(backbone_name, prec):
    backbone_name = backbone_name.lstrip("CLIP-")
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu").eval()

    model = clip.build_model(state_dict or model.state_dict())

    assert prec in ["fp16", "fp32", "amp"]
    if prec == "fp32" or prec == "amp":
        # CLIP's default precision is fp16
        model.float()
  
    return model


def load_vit_to_cpu(backbone_name, prec):
    if backbone_name == "IN21K-ViT-B/16":
        model = vit_base_patch16_224(pretrained=True).eval()
    elif backbone_name == "IN21K-ViT-B/16@384px":
        model = vit_base_patch16_384(pretrained=True).eval()
    elif backbone_name == "IN21K-ViT-L/16":
        model = vit_large_patch16_224(pretrained=True).eval()

    assert prec in ["fp16", "fp32", "amp"]
    if prec == "fp16":
        # ViT's default precision is fp32
        model.half()
    
    return model


class Trainer:
    def __init__(self, cfg):

        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        elif cfg.gpu is None:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(cfg.gpu)
            self.device = torch.device("cuda:{}".format(cfg.gpu))

        self.cfg = cfg
        self.build_data_loader()
        
        self.build_model()
        self.evaluator = Evaluator(cfg)
        self._writer = None


    def build_data_loader(self):
        cfg = self.cfg
        train_p_dataset, train_u_dataset, test_dataset, train_test_dataset = DATASET_GETTERS[cfg.dataset](cfg)

        self.num_classes = cfg.num_classes
        #self.classnames = train_p_dataset.classes
        self.classnames = cfg.classnames
        #self.classnames = self.class_name_binarize(train_p_dataset.classes)

        # self.sampled_cls_num_list = self.cls_num_list
        self.train_p_loader = DataLoader(train_p_dataset, sampler=RandomSampler(train_p_dataset), batch_size=cfg.batch_size, drop_last=True)
        self.train_u_loader = DataLoader(train_u_dataset, sampler=RandomSampler(train_u_dataset), batch_size=cfg.batch_size * self.cfg.mu_u, drop_last=True)
        self.test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=100, num_workers=4, drop_last=False)
        self.train_test_loader = DataLoader(train_test_dataset, sampler=SequentialSampler(train_test_dataset), batch_size=100, num_workers=4, drop_last=False)
        
    def build_model(self):
        cfg = self.cfg
        classnames = self.classnames
        num_classes = self.num_classes

        print("Building model")
        if cfg.zero_shot:
            assert cfg.backbone.startswith("CLIP")
            print(f"Loading CLIP (backbone: {cfg.backbone})")
            clip_model = load_clip_to_cpu(cfg.backbone, cfg.prec)
            self.model = ZeroShotCLIP(clip_model)
            #self.model = ZeroShotCLIPBinary(clip_model, cfg.positive_label_list)
            self.model.to(self.device)
            self.tuner = None
            #self.head = None
            self.head_0 = None
            self.head_1 = None
            template = "a photo of a {}."
            prompts = self.get_tokenized_prompts(classnames, template)
            self.model.init_text_features(prompts)

        elif cfg.backbone.startswith("CLIP"):
            print(f"Loading CLIP (backbone: {cfg.backbone})")
            clip_model = load_clip_to_cpu(cfg.backbone, cfg.prec)
            print('num_classes: {}'.format(num_classes))
            self.model = PeftModelFromCLIP(cfg, clip_model, num_classes)
            self.model.to(self.device)
            self.tuner = self.model.tuner
            #self.head = self.model.head
            self.head_0 = self.model.head_0
            self.head_1 = self.model.head_1

        elif cfg.backbone.startswith("IN21K-ViT"):
            print(f"Loading ViT (backbone: {cfg.backbone})")
            vit_model = load_vit_to_cpu(cfg.backbone, cfg.prec)
            self.model = PeftModelFromViT(cfg, vit_model, num_classes)
            self.model.to(self.device)
            self.tuner = self.model.tuner
            #self.head = self.model.head
            self.head_0 = self.model.head_0
            self.head_1 = self.model.head_1

        if not (cfg.zero_shot or cfg.test_train or cfg.test_only):
            self.build_optimizer()
            self.build_criterion()

            if cfg.init_head == "text_feat":
                self.init_head_text_feat()
            elif cfg.init_head in ["class_mean", "1_shot", "10_shot", "100_shot"]:
                self.init_head_class_mean()
            elif cfg.init_head == "linear_probe":
                self.init_head_linear_probe()
            else:
                print("No initialization with head")
            
            torch.cuda.empty_cache()
        
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1 and cfg.gpu is None:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def build_optimizer(self):
        cfg = self.cfg

        print("Turning off gradients in the model")
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
        print("Turning on gradients in the tuner")
        for name, param in self.tuner.named_parameters():
            param.requires_grad_(True)
        print("Turning on gradients in the head")
        '''
        for name, param in self.head.named_parameters():
            param.requires_grad_(True)
        '''
        for name, param in self.head_0.named_parameters():
            param.requires_grad_(True)
        for name, param in self.head_1.named_parameters():
            param.requires_grad_(True)
        # print parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        tuned_params = sum(p.numel() for p in self.tuner.parameters())
        #head_params = sum(p.numel() for p in self.head.parameters())
        head_0_params = sum(p.numel() for p in self.head_0.parameters())
        head_1_params = sum(p.numel() for p in self.head_1.parameters())

        print(f"Total params: {total_params}")
        print(f"Tuned params: {tuned_params}")
        #print(f"Head params: {head_params}")
        print(f"Head0 params: {head_0_params}")
        print(f"Head1 params: {head_1_params}")
        # for name, param in self.tuner.named_parameters():
        #     print(name, param.numel())

        # NOTE: only give tuner and head to the optimizer
        self.optim = torch.optim.SGD([{"params": self.tuner.parameters()},
                                      #{"params": self.head.parameters()}],
                                      {"params": self.head_0.parameters()},
                                      {"params": self.head_1.parameters()}],
                                      lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, cfg.num_epochs)
        self.scaler = GradScaler() if cfg.prec == "amp" else None

    def build_criterion(self):
        cfg = self.cfg

        if cfg.loss_type == "uPU" or cfg.loss_type == "nnPU":
            self.ure_criterion = PULoss(p_prior=cfg.p_prior, loss_type=cfg.loss_type)

        self.cr_criterion = nn.CrossEntropyLoss(reduction='none', label_smoothing=cfg.ls_weight)
        
    def get_tokenized_prompts(self, classnames, template):
        prompts = [template.format(c.replace("_", " ")) for c in classnames]
        # print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)
        return prompts

    def route_module(self, images, targets_p, targets_u, targets_u_true, epoch_idx):
        cfg = self.cfg
        image_features = self.model(images)
        # outputs of the first head
        outputs_0 = self.head_0(image_features)
        outputs_0 = de_interleave(outputs_0, 2 * cfg.mu_u + 1)
        outputs_p_w_0 = outputs_0[: cfg.batch_size]
        outputs_u_w_0, outputs_u_s_0 = outputs_0[cfg.batch_size: ].chunk(2)

        # outputs of the second head
        outputs_1 = self.head_1(image_features)
        outputs_1 = de_interleave(outputs_1, 2 * cfg.mu_u + 1)
        outputs_p_w_1 = outputs_1[: cfg.batch_size]
        outputs_u_w_1, outputs_u_s_1 = outputs_1[cfg.batch_size: ].chunk(2) 
        ure_loss = self.ure_criterion(outputs_p_w=outputs_p_w_0, outputs_u_w=outputs_u_w_0, targets_p=targets_p, targets_u=targets_u)


        confidence_mat_0 = torch.softmax(outputs_u_w_0.detach(), dim=-1)
        confidence_0, pseudo_label_0 = torch.max(confidence_mat_0, dim=-1)
        mask_0 = confidence_0.ge(cfg.threshold)

        pos_pseudo_label_mask = ((pseudo_label_0==0) * mask_0).to(self.device)
        neg_pseudo_label_mask = ((pseudo_label_0==1) * mask_0).to(self.device)

        pos_total_loss = torch.zeros(1).to(self.device)
        if pos_pseudo_label_mask.float().sum() > 0:
            targets_p_cr = pseudo_label_0[pos_pseudo_label_mask]
            outputs_p_cr = outputs_u_s_1[pos_pseudo_label_mask]
            pos_total_loss = self.cr_criterion(outputs_p_cr,targets_p_cr).mean()
        neg_total_loss = torch.zeros(1).to(self.device)
        if neg_pseudo_label_mask.float().sum() > 0:
            targets_n_cr = pseudo_label_0[neg_pseudo_label_mask]
            outputs_n_cr = outputs_u_s_1[neg_pseudo_label_mask]
            neg_total_loss = self.cr_criterion(outputs_n_cr,targets_n_cr).mean()

        #pos_data_loss = self.cr_criterion(outputs_p_w_1,targets_p).mean()
        cr_loss = cfg.p_prior * pos_total_loss + (1-cfg.p_prior) * neg_total_loss
        #cr_loss = (self.cr_criterion(outputs_u_s_1,pseudo_label_0) * mask_0).mean() * mask_0.shape[0] / (mask_0.sum() + 1e-12)
        #cr_loss = (self.cr_criterion(outputs_u_s_1,targets_u_true) * mask_0).mean() * mask_0.shape[0] / (mask_0.sum() + 1e-12)
        #calculate the accuracy of pseudo-labels
        pseudo_label_acc = ((pseudo_label_0 == targets_u_true) * mask_0).sum() / mask_0.sum() if mask_0.sum() > 0 else 0
        #pseudo_label_acc = (pseudo_label_0 == targets_u_true).sum() / targets_u_true.shape[0]

        if epoch_idx >= cfg.warm_up_epoch:
            loss = ure_loss + cr_loss
        else:
            loss = ure_loss
        return loss, ure_loss, cr_loss, pseudo_label_acc
        #return loss, ure_loss, cr_loss, mask_0.sum()

    def train(self):
        cfg = self.cfg

        # Initialize summary writer
        writer_dir = os.path.join(cfg.output_dir, "tensorboard")
        os.makedirs(writer_dir, exist_ok=True)
        print(f"Initialize tensorboard (log_dir={writer_dir})")
        self._writer = SummaryWriter(log_dir=writer_dir)

        # Initialize average meters
        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_meter = AverageMeter(ema=True)
        acc_meter = AverageMeter(ema=True)

        # newly added
        ure_loss_meter = AverageMeter(ema=True)
        cr_loss_meter = AverageMeter(ema=True)
        pseudo_label_acc_meter = AverageMeter(ema=True)


        # Remember the starting time (for computing the elapsed time)
        time_start = time.time()

        num_epochs = cfg.num_epochs
        for epoch_idx in range(num_epochs):
            p_iter = iter(self.train_p_loader)
            u_iter = iter(self.train_u_loader)
            self.tuner.train()
            #self.head.train()
            self.head_0.train()
            self.head_1.train()
            end = time.time()

            #num_batches = len(self.train_loader)
            #num_batches = len(self.train_u_loader)
            num_batches = cfg.num_iterations_per_epoch
            for batch_idx in range(num_batches):
                #print('batch idx: {}'.format(batch_idx))
                try:
                    (images_p_w, _), targets_p = next(p_iter)
                except:
                    p_iter = iter(self.train_p_loader)
                    (images_p_w, _), targets_p = next(p_iter)
                #print('images_p_w shape: {}'.format(images_p_w.shape))
                try:
                    (images_u_w, images_u_s), (targets_u, targets_u_true) = next(u_iter)
                except:
                    u_iter = iter(self.train_u_loader)
                    (images_u_w, images_u_s), (targets_u, targets_u_true) = next(u_iter)
                #print('images_u_w shape: {}'.format(images_u_w.shape))
                images = interleave(torch.cat((images_p_w, images_u_w, images_u_s)), 2 * cfg.mu_u + 1).to(self.device)
                #print('images shape: {}'.format(images.shape))
                targets_p, targets_u, targets_u_true = targets_p.to(self.device), targets_u.to(self.device), targets_u_true.to(self.device)
                data_time.update(time.time() - end)

                if cfg.prec == "amp":

                    with autocast():
                        loss, ure_loss, cr_loss, pseudo_label_acc = self.route_module(images, targets_p, targets_u, targets_u_true, epoch_idx)
                        self.scaler.scale(loss).backward()
                    self.scaler.step(self.optim)
                    self.scaler.update()
                    self.optim.zero_grad()
                else:
                    loss, ure_loss, cr_loss, pseudo_label_acc = self.route_module(images, targets_p, targets_u, targets_u_true, epoch_idx)
                    loss.backward()
                    self.optim.step()
                    self.optim.zero_grad()

                current_lr = self.optim.param_groups[0]["lr"]
                loss_meter.update(loss.item())
                #acc_meter.update(acc.item())
                ure_loss_meter.update(ure_loss.item())
                cr_loss_meter.update(cr_loss.item())
                if pseudo_label_acc > 0:
                    pseudo_label_acc_meter.update(pseudo_label_acc.item())

                batch_time.update(time.time() - end)


                meet_freq = (batch_idx + 1) % cfg.print_freq == 0
                only_few_batches = num_batches < cfg.print_freq
                if meet_freq or only_few_batches:
                    nb_remain = 0
                    nb_remain += num_batches - batch_idx - 1
                    nb_remain += (
                        num_epochs - epoch_idx - 1
                    ) * num_batches
                    eta_seconds = batch_time.avg * nb_remain
                    eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                    info = []
                    info += [f"epoch [{epoch_idx + 1}/{num_epochs}]"]
                    info += [f"batch [{batch_idx + 1}/{num_batches}]"]
                    info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                    info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                    info += [f"total loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})"]
                    info += [f"ure loss {ure_loss_meter.val:.4f} ({ure_loss_meter.avg:.4f})"]
                    info += [f"cr loss {cr_loss_meter.val:.4f} ({cr_loss_meter.avg:.4f})"]
                    info += [f"pseudo label acc {pseudo_label_acc_meter.val:.4f} ({pseudo_label_acc_meter.avg:.4f})"]
                    #info += [f"acc {acc_meter.val:.4f} ({acc_meter.avg:.4f})"]
                    info += [f"lr {current_lr:.4e}"]
                    info += [f"eta {eta}"]
                    print(" ".join(info))

                n_iter = epoch_idx * num_batches + batch_idx
                self._writer.add_scalar("train/lr", current_lr, n_iter)
                self._writer.add_scalar("train/loss.val", loss_meter.val, n_iter)
                self._writer.add_scalar("train/loss.avg", loss_meter.avg, n_iter)
                #self._writer.add_scalar("train/acc.val", acc_meter.val, n_iter)
                #self._writer.add_scalar("train/acc.avg", acc_meter.avg, n_iter)
                
                end = time.time()

            self.sched.step()
            torch.cuda.empty_cache()
            '''
            print("train test:")
            train_test_acc = self.test(mode="train")
            '''
            print("test:")
            test_acc = self.test()
            self._writer.add_scalar("test/acc", test_acc, epoch_idx)
    
        print("Finish training")
        print("Note that the printed training acc is not precise.",
              "To get precise training acc, use option ``test_train True``.")

        #self.save_feature()
        # show elapsed time
        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Time elapsed: {elapsed}")

        # save model
        #self.save_model(cfg.output_dir)

        #self.test()

        # Close writer
        self._writer.close()

    @torch.no_grad()
    def test(self, mode="test"):
        if self.tuner is not None:
            self.tuner.eval()
        ''' 
        if self.head is not None:
            self.head.eval()
        '''
        if self.head_0 is not None:
            self.head_0.eval()
        if self.head_1 is not None:
            self.head_1.eval()
        self.evaluator.reset()

        if mode == "train":
            print(f"Evaluate on the train set")
            data_loader = self.train_test_loader
        elif mode == "test":
            print(f"Evaluate on the test set")
            data_loader = self.test_loader

        for batch in tqdm(data_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            #output = self.model(image)
            feature = self.model(image)
            #ensemble classifier
            #output = (self.head_0(feature) + self.head_1(feature)) / 2
            #output = self.head_0(feature)
            output = self.head_1(feature)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"test/{k}"
            if self._writer is not None:
                self._writer.add_scalar(tag, v)

        return list(results.values())[0]

    @torch.no_grad()
    def save_feature(self):
        if self.tuner is not None:
            self.tuner.eval()
        if self.head_0 is not None:
            self.head_0.eval()
        if self.head_1 is not None:
            self.head_1.eval()
        self.evaluator.reset()
        self.evaluator.reset()
        print(f"save features on the test set")
        data_loader = self.test_loader
        features = []
        labels = []
        for batch in tqdm(data_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            # output = self.model(image, use_tuner=True, return_feature=True)
            output = self.model(image)
            features.append(output)
            labels.append(label)
        print(output.shape)
        torch.save((features, labels), 'route_test_features.pth')
        print(f"save features on the train set")
        data_loader = self.train_test_loader
        features = []
        labels = []
        for batch in tqdm(data_loader, ascii=True):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            # output = self.model(image, use_tuner=True, return_feature=True)
            output = self.model(image)
            features.append(output)
            labels.append(label)
        print(output.shape)
        torch.save((features, labels), 'route_train_features.pth')
