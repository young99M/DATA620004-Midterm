import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np 
from model import *
import sys
import matplotlib.pyplot as plt
import copy

def get_train_val_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 dataset
        std: std of cifar100 dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    # cifar100_training, cifar100_val = torch.utils.data.random_split(cifar100_training, [len(cifar100_training) - int(0.2 * len(cifar100_training)), int(0.2 * len(cifar100_training))])
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    # cifar100_val_loader = DataLoader(
    #     cifar100_val, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    return cifar100_training_loader # , cifar100_val_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 dataset
        std: std of cifar100 dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = np.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = np.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = np.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = np.mean(data_r), np.mean(data_g), np.mean(data_b)
    std = np.std(data_r), np.std(data_g), np.std(data_b)

    return mean, std


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def get_network(args):
    """ return given network """
    if args.net == 'resnet18':
        net = ResNet18(ResBlock).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    elif args.net == 'shufflenet':
        net = shufflenet().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()
    
    return net



class Cutout:
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for i in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

# for cutmix 
class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    """1.论文里的公式2 求出B的rw,rh"""
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
 
    # uniform
    """2.论文里的公式2 求出B的rx,ry(bbox的中心点)"""
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    #限制坐标区域不超过样本大小
 
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    """3.返回剪裁B区域的坐标值"""
    return bbx1, bby1, bbx2, bby2

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def plot_data_loader_image(data_loader, cifar100_mean, cifar100_std):
    plt.rcParams['figure.figsize'] = (15.0, 15.0)
    plt.subplots_adjust(hspace=0.1,wspace=0.1)
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 3)
    device = 'cuda' if torch.cuda.is_available() else 'gpu'

    for index, data in enumerate(data_loader):
        images, labels = data
        images_1, images_2, images_3, images_4 = copy.deepcopy(images), copy.deepcopy(images), copy.deepcopy(images), copy.deepcopy(images)
    
        if index == 0:
            for i in range(4 * plot_num):
                label = labels[i%plot_num].item()
                plt.subplot(4, plot_num, i+1)
                if i+1 <= plot_num:
                    plt.xlabel(str(label)+', baseline')
                    # [C, H, W] -> [H, W, C]
                    img = images_1[i%plot_num].numpy().transpose(1, 2, 0)
                elif plot_num < i+1 <= 2 * plot_num:
                    plt.xlabel(str(label)+', cutout')
                    cut = Cutout(n_holes=1, length=16)
                    images_2[i%plot_num] = cut(images_2[i%plot_num])
                    # [C, H, W] -> [H, W, C]
                    img = images_2[i%plot_num].numpy().transpose(1, 2, 0)
                elif 2 * plot_num < i+1 <= 3 * plot_num:
                    plt.xlabel(str(label)+', mixup')
                    images_3 = mixup_data(images_3, labels, 1.0, True)[0]
                    # [C, H, W] -> [H, W, C]
                    img = images_3[i%plot_num].numpy().transpose(1, 2, 0)
                else:
                    plt.xlabel(str(label)+', cutmix')
                    lam = np.random.beta(1.0, 1.0)
                    rand_index = torch.randperm(images.size()[0]).to(device)
                    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                    images_4[:, :, bbx1:bbx2, bby1:bby2] = images_4[rand_index, :, bbx1:bbx2, bby1:bby2]
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
                    # [C, H, W] -> [H, W, C]
                    img = images_4[i%plot_num].numpy().transpose(1, 2, 0)
                # 反Normalize操作
                img = (img * cifar100_std + cifar100_mean) * 255 
                plt.xticks([])  # 去掉x轴的刻度
                plt.yticks([])  # 去掉y轴的刻度
                plt.imshow(img.astype('uint8'))
            plt.subplots_adjust(hspace=0.1,wspace=0.1)
            plt.show()
        else:
            break 

if __name__ == '__main__':
    cifar100_mean = [0.5071, 0.4867, 0.4408]
    cifar100_std = [0.2675, 0.2565, 0.2761]
    cifar100_test_loader = get_test_dataloader(cifar100_mean, cifar100_std, num_workers=4, batch_size=64, shuffle=True)
    plot_data_loader_image(cifar100_test_loader, cifar100_mean, cifar100_std)