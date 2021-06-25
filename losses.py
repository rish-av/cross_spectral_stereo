import torch
import torch.nn as nn
import torch.nn.functional as F

def warp(img, disp):
    b, _, h, w = img.size()
    x_base = torch.linspace(0, 1, w).repeat(b, h, 1).type_as(img)
    y_base = torch.linspace(0, 1, h).repeat(b, w, 1).transpose(1, 2).type_as(img)
    x_shifts = disp[:, 0, :, :] / w

    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
    output = F.grid_sample(img, 2 * flow_field - 1, mode='bilinear', padding_mode='border',align_corners=True)

    return output

def gradient_x(img):
    gx = torch.add(img[:,:,:-1,:], -1, img[:,:,1:,:])
    return gx

def gradient_y(img):
    gy = torch.add(img[:,:,:,:-1], -1, img[:,:,:,1:])
    return gy

def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu_x = F.avg_pool2d(x, 3, 1, 0)
    mu_y = F.avg_pool2d(y, 3, 1, 0)
    
    sigma_x  = F.avg_pool2d(x ** 2, 3, 1, 0) - mu_x ** 2
    sigma_y  = F.avg_pool2d(y ** 2, 3, 1, 0) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y , 3, 1, 0) - mu_x * mu_y
    
    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    
    SSIM = SSIM_n / SSIM_d
    
    return torch.clamp((1 - SSIM) / 2, 0, 1)


def ap_loss(alpha,est, img):
    l1_warp2 = torch.abs(est - img)
    l1_reconstruction_loss_warp2 = torch.mean(l1_warp2)
    ssim_warp2 = SSIM(est, img)
    ssim_loss_warp2 = torch.mean(ssim_warp2)
    image_loss_warp2  = alpha * ssim_loss_warp2 + (1 - alpha) * l1_reconstruction_loss_warp2
    return image_loss_warp2


def disparity_smoothness(disp, img):
    img_grad_x = img[:, :, :, :-1] - img[:, :, :, 1:]
    img_grad_y = img[:, :, :-1, :] - img[:, :, 1:, :]
    weight_x = torch.exp(-torch.abs(img_grad_x).mean(1).unsqueeze(1))
    weight_y = torch.exp(-torch.abs(img_grad_y).mean(1).unsqueeze(1))

    loss = (((disp[:, :, :, :-1] - disp[:, :, :, 1:]).abs() * weight_x).sum() +
            ((disp[:, :, :-1, :] - disp[:, :, 1:, :]).abs() * weight_y).sum()) / \
           (weight_x.sum() + weight_y.sum())

    return loss


def lr_consistency_loss(dl,dr):
    lr_consis_l = torch.mean(torch.abs(dl-warp(dr,dl)))
    lr_consis_r = torch.mean(torch.abs(dr-warp(dl,dr)))

    return lr_consis_r + lr_consis_l
