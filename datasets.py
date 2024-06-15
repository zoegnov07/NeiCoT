# --------------------------------------------------------
# Based on MAE, CPC, BEiT, timm, DINO and DeiT code bases
# https://github.com/pengzhiliang/MAE-pytorch
# https://github.com/jefflai108/Contrastive-Predictive-Coding-PyTorch
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import os
import torch

from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

from timm.data import create_transform

from masking_generator import RandomMaskingGenerator
from dataset_folder import ImageFolder
# hyperspectral
import random
import numpy as np
from scipy.io import loadmat
from sklearn import preprocessing

class DataAugmentationForMAE(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        self.transform = transforms.Compose([       # transformer内置给定的一些初始参数
            transforms.RandomResizedCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        self.masked_position_generator = RandomMaskingGenerator(
            args.window_size, args.mask_ratio
        )

    def __call__(self, image):      # 不执行
        return self.transform(image), self.masked_position_generator()

    def __repr__(self):     # 有print这个类就会执行这
        repr = "(DataAugmentationForBEiT,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_pretraining_dataset(args):
    transform = DataAugmentationForMAE(args)
    print("Data Aug = %s" % str(transform))     # print(transform) 等同于执行 print(transform.__repr__())
    return ImageFolder(args.data_path, transform=transform)     # ImageFolder假设所有的文件按文件夹保存，每个文件夹下存储同一个类别的图片
    

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        if args.crop_pct is None:
            if args.input_size < 384:
                args.crop_pct = 224 / 256
            else:
                args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

# Hyperspectral
DATASETS_CONFIG = {
    'PaviaU': {
        'img': 'PaviaU.mat',
        'gt': 'PaviaU_gt.mat'
    },
    'KSC': {
        'img': 'KSC.mat',
        'gt': 'KSC_gt.mat'
    },
    'IndianPines': {
        'img': 'indian_pines.mat',
        'gt': 'indian_pines_gt.mat'
    },
    'Salinas': {
        'img': 'salinas.mat',
        'gt': 'salinas_gt.mat'
    },
    'Houston': {
        'img': 'Houston.mat',
        'gt': 'Houston_gt.mat'
    },
    'Botswana': {
        'img': 'Botswana.mat',
        'gt': 'Botswana_gt.mat'
    }
}

def get_dataset(dataset_name, target_folder="./dataset/", datasets=DATASETS_CONFIG):
    if dataset_name not in datasets.keys():
        raise ValueError("{} dataset is unknown.".format(dataset_name))

    img_file = target_folder + dataset_name + '/' + datasets[dataset_name].get('img')
    # img_file的输出是：../dataset/IndianPines/indian_pines.mat
    gt_file = target_folder + dataset_name + '/' + datasets[dataset_name].get('gt')
    # gt_file的输出是：../dataset/IndianPines/indian_pines_gt.mat
    if dataset_name == 'Houston':
        # Load the image
        img = loadmat(img_file)['Houston']
        gt = loadmat(gt_file)['Houston_gt']
        label_values = ["Undefined", "Healthy grass", "Stressed grass", " Synthetic grass",
                        "Trees", "Soil", "Water", "Residential", "Commercial", "Road",
                        "Highway", "Railway", "Parking Lot 1", "Parking Lot 2",
                        "Tennis Court", "Running Track"]
        ignored_labels = [0]

    elif dataset_name == 'PaviaU':
        # Load the image
        img = loadmat(img_file)['paviaU']
        gt = loadmat(gt_file)['Data_gt']
        label_values = ['Undefined', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
                        'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']
        ignored_labels = [0]

    elif dataset_name == 'IndianPines':
        # Load the image
        img = loadmat(img_file)
        img = img['HSI_original']
        gt = loadmat(gt_file)['Data_gt']
        # print(gt.shape)  输出是：(145，145)
        label_values = ["Undefined", "Alfalfa", "Corn-notill", "Corn-mintill",
                        "Corn", "Grass-pasture", "Grass-trees",
                        "Grass-pasture-mowed", "Hay-windrowed", "Oats",
                        "Soybean-notill", "Soybean-mintill", "Soybean-clean",
                        "Wheat", "Woods", "Buildings-Grass-Trees-Drives",
                        "Stone-Steel-Towers"]
        ignored_labels = [0]

    elif dataset_name == 'Botswana':
        # Load the image
        img = loadmat(img_file)['Botswana']

        gt = loadmat(gt_file)['Botswana_gt']
        label_values = ["Undefined", "Water", "Hippo grass",
                        "Floodplain grasses 1", "Floodplain grasses 2",
                        "Reeds", "Riparian", "Firescar", "Island interior",
                        "Acacia woodlands", "Acacia shrublands",
                        "Acacia grasslands", "Short mopane", "Mixed mopane",
                        "Exposed soils"]
        ignored_labels = [0]

    elif dataset_name == 'KSC':
        # Load the image
        img = loadmat(img_file)['KSC']

        gt = loadmat(gt_file)['KSC_gt']
        label_values = ["Undefined", "Scrub", "Willow swamp",
                        "Cabbage palm hammock", "Cabbage palm/oak hammock",
                        "Slash pine", "Oak/broadleaf hammock",
                        "Hardwood swamp", "Graminoid marsh", "Spartina marsh",
                        "Cattail marsh", "Salt marsh", "Mud flats", "Wate"]
        ignored_labels = [0]

    elif dataset_name == 'Salinas':
        # Load the image
        img = loadmat(img_file)['HSI_original']

        gt = loadmat(gt_file)['Data_gt']
        label_values = ["Undefined", "Brocoli green weeds 1", "Brocoli_green_weeds_2",
                        "Fallow", "Fallow rough plow", "Fallow smooth", "Stubble",
                        "Celery", "Grapes untrained", "Soil vinyard develop",
                        "Corn senesced green weeds", "Lettuce romaine 4wk",
                        "Lettuce romaine 5wk", "Lettuce romaine 6wk", "Lettuce romaine 7wk",
                        "Vinyard untrained", "Vinyard vertical trellis"]
        ignored_labels = [0]
    elif dataset_name == 'WHU-Hi-HanChuan':
        # Load the image
        img = loadmat(img_file)['WHU_Hi_HanChuan']

        gt = loadmat(gt_file)['WHU_Hi_HanChuan_gt']
        label_values = ["Undefined", "Strawberry", "Cowpea",
                        "Soybean", "Sorghum", "Water spinach", "Watermelon",
                        "Greens", "Trees", "Grass",
                        "Red roof", "Gray roof",
                        "Plastic", "Bare soil", "Road",
                        "Bright object", "Water"]

        ignored_labels = [0]

    # elif dataset_name == 'HyRANK':
    #     # Load the image
    #     img = loadmat(img_file)['Dioni']
    #     gt = loadmat(gt_file)['Dioni_GT']
    #     label_values = ["Undefined", "Dense urban fabric", "Mineral extraction site",
    #                     "Non-irrigated arable land", "Fruit trees", "Olive groves", "Broad-leaved forest",
    #                     "Coniferous forest", "Mixed forest", "Dense sclerophyllous vegetation",
    #                     "Sparce sclerophyllous vegetation", "Sparsely vegetated areas",
    #                     "Rocks and sand", "Water", "Coastal water"]

    #     ignored_labels = [0]
    
    # 检查像元是否为空
    nan_mask = np.isnan(img.sum(axis=-1))   # axis=-1：在最后一维操作,将200维的波段求和
    # if np.count_nonzero(nan_mask) > 0:
    #     logger.info("Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN "
    #           "data is disabled.")
    img[nan_mask] = 0
    gt[nan_mask] = 0
    ignored_labels.append(0)

    ignored_labels = list(set(ignored_labels))  # set() 函数是将字符串、列表、元组、range 对象等可迭代对象转换成集合。
    # Normalization
    img = np.asarray(img, dtype='float32')  # 将结构数据转化为ndarray。
    data = img.reshape(np.prod(img.shape[:2]), np.prod(img.shape[2:]))  # 将145，145，200的三维数组转为了145x145，200的二维数组。
    data = preprocessing.minmax_scale(data, axis=1) # preprocessing.minmax_scale()将每个特征放缩到给定范围内（默认范围0-1），也就是归一化。(21025,200)
    scaler = preprocessing.StandardScaler() # 数据标准化
    scaler.fit(data)    # 计算输入数据各特征的平均值，标准差以及之后的缩放系数
    data = scaler.fit_transform(data)   # fit_transform通过fit_params调整数据,使得每个特征的数据分布平均值为0，方差为1
    img = data.reshape(img.shape) # 转化为了原来数据的形状
    return img, gt, label_values, ignored_labels  # label_values, ignored_labels

class HyperX(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperX, self).__init__()
        self.data = data        # data.shape: (145,145,200)
        self.label = gt         # gt.shape:  (145,145)
        self.dataset_name = hyperparams['dataset']      # IndianPines
        self.patch_size = hyperparams['patch_size']     # 11
        self.ignored_labels = set(hyperparams['ignored_labels'])        # {0}
        self.center_pixel = hyperparams['center_pixel']     # True
        supervision = hyperparams['supervision']        # full
        self.n_bands = hyperparams['n_bands']            # 高光谱的波段维度200
        self.mask_ratio = hyperparams['mask_ratio']
        self.control_patch = hyperparams['control_patch']
        self.random_mask = hyperparams['random_mask']
        self.gauss_std = hyperparams['gauss_std']
        # print('这里是现在要看的地方：',supervision)
        # Fully supervised : use all pixels with label not ignored
        if supervision == 'full':
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        # Semi-supervised : use all pixels, except padding
        elif supervision == 'semi':
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)     # mask是145*145，返回给非零是有两个维度，分别赋给x_pos, y_pos
        p = self.patch_size // 2        # 5
        # 5<x<140 and 5<y<140， 没有填充，防止取到外面的值，根据patch来确定范围。
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos) if
                                 x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])
        self.labels = [self.label[x, y] for x, y in self.indices]
                
        np.random.shuffle(self.indices)

    @staticmethod
    def get_data(data, x, y, patch_size, data_3D=False):
        x1, y1 = x - patch_size // 2, y - patch_size // 2
        x2, y2 = x1 + patch_size, y1 + patch_size
        data = data[x1:x2, y1:y2]
        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        # Add a fourth dimension for 3D CNN
        if data_3D:
            # Make 4D data ((Batch x) Planes x Channels x Width x Height)
            data = data.unsqueeze(0)  # uncommon if need.
        # print("data.shape_next",data.shape)       # (200,1111)
        return data

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        data = self.get_data(self.data, x, y, self.patch_size, data_3D=False)
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size
        label = self.label[x1:x2, y1:y2]
        label = np.asarray(np.copy(label), dtype='int64')
        label = torch.from_numpy(label)
        
        #对波段上进行掩码,并且返回mask
        # patch = int(self.n_bands / self.control_patch)
        # num_mask = int(self.mask_ratio * patch)
        # mask = np.hstack([
        #         np.zeros(patch - num_mask),
        #         np.ones(num_mask),
        #     ])
        # 对像素进行掩码
        patch = int(self.patch_size ** 2)
        num_mask = int(self.mask_ratio * patch)
        mask = np.hstack([
                np.zeros(patch - num_mask),
                np.ones(num_mask),
            ])
        if self.random_mask == "True":
            np.random.shuffle(mask)
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        # mask_band = self.masked_position_generator(self)
        # print("data.shape",data.shape)
        
        # 添加高斯噪声
        std = self.gauss_std
        data_std = torch.from_numpy(np.random.normal(0, std, size=data.size())).float()
        data = data + data_std

        return data, label, mask
    
    
    
class HyperF(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperF, self).__init__()
        self.data = data        # data.shape: (145,145,200)
        self.label = gt         # gt.shape:  (145,145)
        self.dataset_name = hyperparams['dataset']      # IndianPines
        self.patch_size = hyperparams['patch_size']     # 11
        self.ignored_labels = set(hyperparams['ignored_labels'])        # {0}
        self.center_pixel = hyperparams['center_pixel']     # True
        supervision = hyperparams['supervision']        # full
        # self.n_bands = hyperparams['n_bands']            # 高光谱的波段维度200
        self.gauss_std = hyperparams['gauss_std']
        # print('这里是现在要看的地方：',supervision)
        # Fully supervised : use all pixels with label not ignored
        if supervision == 'full':
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        # Semi-supervised : use all pixels, except padding
        elif supervision == 'semi':
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)     # mask是145*145，返回给非零是有两个维度，分别赋给x_pos, y_pos
        p = self.patch_size // 2        # 5
        # 5<x<140 and 5<y<140， 没有填充，防止取到外面的值，根据patch来确定范围。
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos) if
                                 x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])
        self.labels = [self.label[x, y] for x, y in self.indices]
                
        np.random.shuffle(self.indices)

    @staticmethod
    def get_data(data, x, y, patch_size, data_3D=False):
        x1, y1 = x - patch_size // 2, y - patch_size // 2
        x2, y2 = x1 + patch_size, y1 + patch_size
        data = data[x1:x2, y1:y2]
        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        # Add a fourth dimension for 3D CNN
        if data_3D:
            # Make 4D data ((Batch x) Planes x Channels x Width x Height)
            data = data.unsqueeze(0)  # uncommon if need.
        # print("data.shape_next",data.shape)       # (200,1111)
        return data

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        data = self.get_data(self.data, x, y, self.patch_size, data_3D=False)
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size
        label = self.label[x1:x2, y1:y2]
        label = np.asarray(np.copy(label), dtype='int64')
        label = torch.from_numpy(label)        
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        
        # 添加高斯噪声
        std = self.gauss_std
        data_std = torch.from_numpy(np.random.normal(0, std, size=data.size())).float()
        data = data + data_std
        
        # data_std = torch.from_numpy(np.random.normal(1, std, size=data.size())).float()
        # data = data * data_std

        return data, label



class HyperV(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperV, self).__init__()
        self.data = data        # data.shape: (145,145,200)
        self.label = gt         # gt.shape:  (145,145)
        self.dataset_name = hyperparams['dataset']      # IndianPines
        self.patch_size = hyperparams['patch_size']     # 11
        self.ignored_labels = set(hyperparams['ignored_labels'])        # {0}
        self.center_pixel = hyperparams['center_pixel']     # True
        supervision = hyperparams['supervision']        # full
        # self.n_bands = hyperparams['n_bands']            # 高光谱的波段维度200
        # print('这里是现在要看的地方：',supervision)
        # Fully supervised : use all pixels with label not ignored
        if supervision == 'full':
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        # Semi-supervised : use all pixels, except padding
        elif supervision == 'semi':
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)     # mask是145*145，返回给非零是有两个维度，分别赋给x_pos, y_pos
        p = self.patch_size // 2        # 5
        # 5<x<140 and 5<y<140， 没有填充，防止取到外面的值，根据patch来确定范围。
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos) if
                                 x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])
        self.labels = [self.label[x, y] for x, y in self.indices]
                
        np.random.shuffle(self.indices)

    @staticmethod
    def get_data(data, x, y, patch_size, data_3D=False):
        x1, y1 = x - patch_size // 2, y - patch_size // 2
        x2, y2 = x1 + patch_size, y1 + patch_size
        data = data[x1:x2, y1:y2]
        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        # Add a fourth dimension for 3D CNN
        if data_3D:
            # Make 4D data ((Batch x) Planes x Channels x Width x Height)
            data = data.unsqueeze(0)  # uncommon if need.
        # print("data.shape_next",data.shape)       # (200,1111)
        return data

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        data = self.get_data(self.data, x, y, self.patch_size, data_3D=False)
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size
        label = self.label[x1:x2, y1:y2]
        label = np.asarray(np.copy(label), dtype='int64')
        label = torch.from_numpy(label)        
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]

        return data, label