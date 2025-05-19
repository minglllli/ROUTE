from collections import defaultdict
import numpy as np
import torchvision
from torchvision import datasets,transforms

from .randaugment import RandAugmentMC
from PIL import Image
import json
from torch.utils.data import Dataset
def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)
#eurosat_mean = (0.3445, 0.3805, 0.4081)
#eurosat_std = (0.0932, 0.0649, 0.0543)
#clip_mean=(0.48145466, 0.4578275, 0.40821073)
#clip_std=(0.26862954, 0.26130258, 0.27577711)
clip_mean = (0.4792, 0.4471, 0.3974)
clip_std = (0.2312, 0.2279, 0.2298)
def get_oxfordpets(cfg):
    resize_dim = 256
    crop_dim = 224


    transform_val = transforms.Compose([
        transforms.Resize((crop_dim, crop_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=clip_mean, std=clip_std)])

    data_t = read_json(cfg.split_root)
    data_t = data_t['train']
    targets = []
    for i in range(len(data_t)):
        targets.append(data_t[i][1])
    if cfg.generation_setting == "one_sample":
        train_positive_idxs, train_unlabeled_idxs = p_u_split_one_sample(cfg, targets )
    elif cfg.generation_setting == "two_sample":
        train_positive_idxs, train_unlabeled_idxs = p_u_split_two_sample(cfg, targets )
    
    train_p_dataset = Oxfordpets(
        cfg.root, cfg.split_root, train_positive_idxs, train=True,
        transform=TransformFixMatch(mean=clip_mean, std=clip_std),
        target_transform=TransformPTarget(positive_label_list=cfg.positive_label_list))

    train_u_dataset = Oxfordpets(
        cfg.root, cfg.split_root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=clip_mean, std=clip_std),
        target_transform=TransformUTarget(positive_label_list=cfg.positive_label_list)
        )

    test_dataset = Oxfordpets(
        cfg.root, cfg.split_root, train=False, transform=transform_val,
        target_transform=TransformTestTarget(positive_label_list=cfg.positive_label_list))

    train_test_dataset = Oxfordpets(
        cfg.root, cfg.split_root, train_unlabeled_idxs, train=True,
        transform=transform_val,
        target_transform=TransformTestTarget(positive_label_list=cfg.positive_label_list)
        )
    '''
    train_p_dataset = CIFAR100SSL(
        cfg.root, train_positive_idxs, train=True,
        transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std)
        )

    train_u_dataset = CIFAR100SSL(
        cfg.root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std)
        )

    test_dataset = datasets.CIFAR100(
        cfg.root, train=False, transform=transform_val, download=False
        )
    '''
    return train_p_dataset, train_u_dataset, test_dataset, train_test_dataset
# one-sample setting
def p_u_split_one_sample(cfg, labels):
    #print('available classes:', cfg.positive_label_list, 'positive classes:', cfg.positive_label_list, 'number of class:', cfg.num_classes)
    label_per_class = cfg.num_p // len(cfg.positive_label_list)
    labels = np.array(labels)
    positive_idx = []
    # unlabeled data:
    unlabeled_idx = np.array(range(len(labels)))
    for i in cfg.positive_label_list:
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        positive_idx.extend(idx)
    positive_idx = np.array(positive_idx)
    unlabeled_idx = np.setdiff1d(unlabeled_idx, positive_idx)
    assert len(positive_idx) == cfg.num_p
    '''
    if cfg.num_labeled < cfg.batch_size:
        num_expand_x = math.ceil(
            cfg.batch_size * cfg.eval_step / cfg.num_labeled)
        positive_idx = np.hstack([positive_idx for _ in range(num_expand_x)])
    '''
    np.random.shuffle(positive_idx)
    np.random.shuffle(unlabeled_idx)
    return positive_idx, unlabeled_idx

# two-sample setting
def p_u_split_two_sample(cfg, labels):
    labels = np.array(labels)
    positive_idx = []
    unlabeled_idx = []
    pos_per_class = cfg.num_p // len(cfg.positive_label_list)
    unlabel_per_class = cfg.num_u // 37 #eurosat
    for i in range(37):
        idx = np.where(labels == i)[0]
        if i in cfg.positive_label_list:
            idx = np.random.choice(idx, pos_per_class + unlabel_per_class, False)
            positive_idx.extend(idx[:pos_per_class])
            unlabeled_idx.extend(idx[pos_per_class:])
        else:
            idx = np.random.choice(idx, unlabel_per_class, False)
            unlabeled_idx.extend(idx)
    positive_idx = np.array(positive_idx)
    unlabeled_idx = np.array(unlabeled_idx)
    assert len(positive_idx) == cfg.num_p
    assert len(unlabeled_idx) == cfg.num_u
    np.random.shuffle(positive_idx)
    np.random.shuffle(unlabeled_idx)
    return positive_idx, unlabeled_idx

class TransformFixMatch(object):
    def __init__(self, mean, std, img_size=32):
        resize_dim = 256
        crop_dim = 224
        self.weak = transforms.Compose([
            transforms.Resize(resize_dim),
            transforms.RandomCrop(size=crop_dim,
                                  padding=int(crop_dim * 0.125),
                                  padding_mode='reflect'),
            transforms.RandomHorizontalFlip()])
        self.strong = transforms.Compose([
            transforms.Resize(resize_dim),
            transforms.RandomCrop(size=crop_dim,
                                  padding=int(crop_dim*0.125),
                                  padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        #strong1 = self.strong(x)
        #return self.normalize(weak), self.normalize(strong), self.normalize(strong1)
        return self.normalize(weak), self.normalize(strong)

class TransformPTarget(object):
    def __init__(self, positive_label_list):
        self.p_transform = lambda x: 0 if x in positive_label_list else 1

    def __call__(self, x):
        x = self.p_transform(x)
        if x == 1:
            raise NotImplementedError("Error when generating positive data")

        return x


class TransformUTarget(object):
    def __init__(self, positive_label_list):
        self.u_transform = lambda x: 1
        self.true_transform = lambda x: 0 if x in positive_label_list else 1

    def __call__(self, x):
        x_unlabeled = self.u_transform(x)
        x_true = self.true_transform(x)
        return x_unlabeled, x_true


class TransformTestTarget(object):
    def __init__(self, positive_label_list):
        self.true_transform = lambda x: 0 if x in positive_label_list else 1

    def __call__(self, x):
        x = self.true_transform(x)
        return x

class Oxfordpets(Dataset):
    def __init__(self, root, split_root, indexs=None,  train=True,
                 transform=None, target_transform=None):
        data_t = read_json(split_root)
        if train:
            data_t = data_t['train']
        else:
            data_t = data_t['test']
        self.root = root
        self.data = []
        self.targets = []
        for i in range(len(data_t)):
            self.data.append(self.root+'/'+data_t[i][0])
            self.targets.append(data_t[i][1])
        self.transform = transform
        self.target_transform = target_transform
        #print(indexs)
        if indexs is not None:
            self.data = np.array(self.data)[indexs]
            self.targets = np.array(self.targets)[indexs]
        image_data = []
        for add in self.data:
            img = Image.open(add).convert('RGB')
            image_data.append(img)
        self.data = image_data


    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data) 
