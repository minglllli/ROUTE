import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from .clip_text import CLIP_Text
from .peft_vit import Peft_ViT, ViT_Tuner
from .peft_rn import Peft_RN, RN_Tuner
from .classifiers import *


class ZeroShotCLIP(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.text_encoder = CLIP_Text(clip_model)
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale.exp()
        self.dtype = clip_model.dtype

    def encode_text(self, text):
        try:
            text_features = self.text_encoder(text)
        except:
            # CUDA out of memory
            text_split = torch.split(text, 1000)
            text_features = torch.cat([self.text_encoder(x) for x in text_split])
        return text_features

    def encode_image(self, image):
        return self.image_encoder(image.to(self.dtype))
    
    @torch.no_grad()
    def init_text_features(self, prompts):
        text_features = self.encode_text(prompts)
        text_features = F.normalize(text_features, dim=-1)
        self.text_features = text_features
    
    def forward(self, image):
        image_features = self.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)
        logit = self.logit_scale * F.linear(image_features, self.text_features)
        return logit
'''
class ZeroShotCLIPBinary(nn.Module):
    def __init__(self, clip_model, positive_label_list):
        super().__init__()
        self.text_encoder = CLIP_Text(clip_model)
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale.exp()
        self.dtype = clip_model.dtype
        self.positive_label_list = np.array(positive_label_list)

    def encode_text(self, text):
        try:
            text_features = self.text_encoder(text)
        except:
            # CUDA out of memory
            text_split = torch.split(text, 1000)
            text_features = torch.cat([self.text_encoder(x) for x in text_split])
        return text_features

    def encode_image(self, image):
        return self.image_encoder(image.to(self.dtype))
    
    @torch.no_grad()
    def init_text_features(self, prompts):
        text_features = self.encode_text(prompts)
        text_features = F.normalize(text_features, dim=-1)
        #self.text_features = torch.zeros((2, text_features.shape[1])).to(self.dtype)
        pos_text_embedding = text_features[self.positive_label_list].mean(axis=0).unsqueeze(0)
        all_idx = np.arange(text_features.shape[0])
        self.negative_label_list = np.setdiff1d(all_idx, self.positive_label_list)
        neg_text_embedding = text_features[self.negative_label_list].mean(axis=0).unsqueeze(0)
        self.text_features = torch.cat((pos_text_embedding, neg_text_embedding))
        print("text shape: {}".format(self.text_features.shape))
    
    def forward(self, image):
        image_features = self.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)
        logit = self.logit_scale * F.linear(image_features, self.text_features)
        return logit
'''

class PeftModelFromCLIP(nn.Module):
    def __init__(self, cfg, clip_model, num_classes):
        super().__init__()

        if cfg.backbone.startswith("CLIP-ViT"):
            self.text_encoder = CLIP_Text(clip_model)
            self.image_encoder = Peft_ViT(clip_model.visual)
            self.tuner = ViT_Tuner(cfg, clip_model.visual, num_classes)
        elif cfg.backbone.startswith("CLIP-RN"):
            self.text_encoder = CLIP_Text(clip_model)
            self.image_encoder = Peft_RN(clip_model.visual)
            self.tuner = RN_Tuner(cfg, clip_model.visual, num_classes)
        
        feat_dim = self.image_encoder.out_dim
        dtype = self.image_encoder.dtype
        self.head_0 = eval(cfg.classifier)(feat_dim, num_classes, dtype)
        self.head_1 = eval(cfg.classifier)(feat_dim, num_classes, dtype)

    def encode_text(self, text):
        try:
            text_features = self.text_encoder(text)
        except:
            # CUDA out of memory
            text_split = torch.split(text, 1000)
            text_features = torch.cat([self.text_encoder(x) for x in text_split])
        return text_features
    
    def forward(self, image, use_tuner=True, return_feature=False):
        tuner = self.tuner if use_tuner else None
        '''
        head = self.head if not return_feature else None
        return self.image_encoder(image, tuner, head)
        '''
        return self.image_encoder(image, tuner, head=None)



class PeftModelFromViT(nn.Module):
    def __init__(self, cfg, vit_model, num_classes):
        super().__init__()

        if cfg.backbone.startswith("IN21K-ViT"):
            self.image_encoder = Peft_ViT(vit_model)
            self.tuner = ViT_Tuner(cfg, vit_model, num_classes)

        feat_dim = self.image_encoder.out_dim
        dtype = self.image_encoder.dtype
        self.head_0 = eval(cfg.classifier)(feat_dim, num_classes, dtype)
        self.head_1 = eval(cfg.classifier)(feat_dim, num_classes, dtype)

    def forward(self, image, use_tuner=True, return_feature=False):
        tuner = self.tuner if use_tuner else None
        #head = self.head if not return_feature else None
        return self.image_encoder(image, tuner, head=None)
