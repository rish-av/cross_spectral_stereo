from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn.functional as F
import numpy as np
import datetime
from os.path import join


def get_summary_writer(rootdir):
    return SummaryWriter(join(rootdir,get_log_dir()))


def get_log_dir():
    '''
    New log dir at every run according to the time at that point in time.
    '''
    now = datetime.datetime.now()
    return "logs/run-%d-%d-%d-%d-%d-%d"%(now.year,now.month,now.day,now.hour,now.minute,now.second)

def pyramid(im, n_levels=4, anti_aliasing=False):
    _, _, height, width = im.size()
    ims = [im]
    for i in range(1, n_levels):
        h = height // (2 ** i)
        w = width // (2 ** i)
        if anti_aliasing:
            im = gaussian(im)
        resized = F.interpolate(im, (h, w), mode='bilinear')
        ims.append(resized)
    return ims


def warp(im, disp):
    theta = torch.Tensor(np.array([[1, 0, 0], [0, 1, 0]])).cuda()
    theta = theta.expand((disp.size()[0], 2, 3)).contiguous()
    grid = F.affine_grid(theta, disp.size(),align_corners=True)
    disp = disp.transpose(1, 2).transpose(2, 3)
    disp = torch.cat((disp, torch.zeros(disp.size()).cuda()), 3)
    grid = grid + 2 * disp
    sampled = F.grid_sample(im, grid,align_corners=True)
    return sampled


def warp_pyramid(ims, disps, sgn):
    result = []
    for i, im in enumerate(ims):
        disp = sgn * disps[i]
        result.append(warp(im.cuda(), disp.cuda()))
    return result



def fliplr(im):
    return torch.flip(im, [3,])


def fliplr_pyramid(ims):
    result = [fliplr(im) for im in ims]
    return result


def l1_loss(a, b):
    return (a - b).abs()


def l1_mean(im):
    return im.abs().mean(1, keepdim=True)


def sobel(im):
    c = im.size()[1]
    fx = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    fx = fx.view(1, 1, 3, 3).expand(1, c, 3, 3)
    fy = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    fy = fy.view(1, 1, 3, 3).expand(1, c, 3, 3)
    if im.is_cuda:
        fx = fx.cuda()
        fy = fy.cuda()
    gradx = F.pad(F.conv2d(im, fx), (1, 1, 1, 1))
    grady = F.pad(F.conv2d(im, fy), (1, 1, 1, 1))
    return gradx, grady


def gaussian(im):
    smooth = 1/16 * im[:, :, :-2,  :-2] + 1/8 * im[:, :, 1:-1,  :-2] + 1/16 * im[:, :, 2:,  :-2] + \
             1/8  * im[:, :, :-2, 1:-1] + 1/4 * im[:, :, 1:-1, 1:-1] + 1/8  * im[:, :, 2:, 1:-1] + \
             1/16 * im[:, :, :-2, 2:  ] + 1/8 * im[:, :, 1:-1, 2:  ] + 1/16 * im[:, :, 2:, 2:  ]
    smooth = F.pad(smooth, (1, 1, 1, 1), mode='replicate')
    return smooth


def grad(im):
    gradx = F.pad((im[:, :, :, 2:] - im[:, :, :, :-2]) / 2.0, (1, 1, 0, 0))
    grady = F.pad((im[:, :, 2:, :] - im[:, :, :-2, :]) / 2.0, (0, 0, 1, 1))
    return gradx, grady


def dssim(im_a, im_b, ksize=5):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_a = F.avg_pool2d(im_a, ksize, 1)
    mu_b = F.avg_pool2d(im_b, ksize, 1)
    sigma_a = F.avg_pool2d(im_a ** 2, ksize, 1) - mu_a ** 2
    sigma_b = F.avg_pool2d(im_b ** 2, ksize, 1) - mu_b ** 2
    sigma_ab = F.avg_pool2d(im_a * im_b, ksize, 1) - mu_a * mu_b
    SSIM_n = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    SSIM_d = (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a + sigma_b + C2)
    SSIM = SSIM_n / SSIM_d
    pad = ksize // 2
    dssim = F.pad(torch.clamp((1 - SSIM) / 2.0, 0, 1), (pad, pad, pad, pad))
    return dssim


def _split(joint_dataset,val_percent, sampler=SubsetRandomSampler,random_seed=42):

    '''
    function useful if there is no explicit validation scripts available
    joint_dataset = train + val items
    returns train and validation samplers based on sampling strategy
    '''
    dataset_size = len(joint_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_percent * dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    return train_sampler, valid_sampler
