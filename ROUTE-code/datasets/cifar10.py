from collections import defaultdict
import numpy as np
import torchvision
from torchvision import datasets,transforms
from .randaugment import RandAugmentMC
from PIL import Image

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

def get_cifar10(cfg):
    resize_dim = 256
    crop_dim = 224
    '''
    transform_labeled = transforms.Compose([
        transforms.Resize(resize_dim),
        transforms.RandomCrop(size=crop_dim,
                              padding=int(crop_dim * 0.125),
                              padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)])
    '''
    
    transform_val = transforms.Compose([
        transforms.Resize((crop_dim, crop_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)])
    
    base_dataset = datasets.CIFAR10(cfg.root, train=True, download=True)
    if cfg.generation_setting == "one_sample":
        train_positive_idxs, train_unlabeled_idxs = p_u_split_one_sample(cfg, base_dataset.targets)
    '''
    train_p_dataset = CIFAR10SSL(
        cfg.root, train_positive_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std),
        target_transform=TransformPTarget(positive_label_list=cfg.positive_label_list))

    train_u_dataset = CIFAR10SSL(
        cfg.root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std),
        target_transform=TransformUTarget(positive_label_list=cfg.positive_label_list)
        )

    test_dataset = datasets.CIFAR10(
        cfg.root, train=False, transform=transform_val, download=False,
        target_transform=TransformTestTarget(positive_label_list=cfg.positive_label_list))
    '''
    train_p_dataset = CIFAR10SSL(
        cfg.root, train_positive_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std)
        )

    train_u_dataset = CIFAR10SSL(
        cfg.root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std)
        )

    test_dataset = datasets.CIFAR10(
        cfg.root, train=False, transform=transform_val, download=False
        )

    return train_p_dataset, train_u_dataset, test_dataset
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

class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


'''
class CIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_factor=None, rand_number=0, train=True,
                 transform=None, target_transform=None, download=True):
        super().__init__(root, train, transform, target_transform, download)

        if train and imb_factor is not None:
            np.random.seed(rand_number)
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_factor)
            self.gen_imbalanced_data(img_num_list)

        self.classnames = self.classes
        self.labels = self.targets
        self.cls_num_list = self.get_cls_num_list()
        self.num_classes = len(self.cls_num_list)
        
    def get_img_num_per_cls(self, cls_num, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        counter = defaultdict(int)
        for label in self.labels:
            counter[label] += 1
        labels = list(counter.keys())
        labels.sort()
        cls_num_list = [counter[label] for label in labels]
        return cls_num_list


class CIFAR100(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=None, train=train, transform=transform)


class CIFAR100_IR10(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.1, train=train, transform=transform)


class CIFAR100_IR50(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.02, train=train, transform=transform)


class CIFAR100_IR100(IMBALANCECIFAR100):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, imb_factor=0.01, train=train, transform=transform)
'''