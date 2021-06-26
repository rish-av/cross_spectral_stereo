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


def multi_scale_loss(config,imgl,imgr,dispsl,dispsr):
        weights = config.multiscale_disp_weights
        net_loss = 0.
        for weight,displ,dispr in zip(weights,dispsl,dispsr):

            _,_,h,w = displ.shape

            resized_imgl = F.interpolate(imgl,size=(h,w))
            resize_imgr = F.interpolate(imgr,size=(h,w))

            warped_lfromr = warp(resized_imgr,displ)
            warped_rfroml = warp(resize_imgl,dispr)

            ap_loss_1 = ap_loss(config.ap_alpha,warped_lfromr,resize_imgl)
            ap_loss_2  = ap_loss(config.ap_alpha,warped_rfroml,resized_imgr)

            smooth_loss_1 = disparity_smoothness(displ,resize_imgl)
            smooth_loss_2 = disparity_smoothness(dispr,resized_imgr)

            lr_loss1 = lr_consistency_loss(dispr,displ)
            lr_loss2 = lr_consistency_loss(displ,dispr)

            loss = config.alpha_ap*(ap_loss_1 + ap_loss_2) + config.alpha_ds*(smooth_loss_1 + smooth_loss_2) + config.alpha_lr*(lr_loss1 + lr_loss2)
            net_loss+= weight*loss

        return net_loss


def auxilary_loss(real_A,real_B,fake_A,fake_B,displ,dispr):

    warped_A = warp(real_A,dispr)
    warped_B = warp(real_B,displ)

    l2_loss = nn.MSELoss()

    net_loss = l2_loss(warped_A,fake_A) + l2_loss(warped_B,fake_B)


    return net_loss


class GANLoss(nn.Module):
    def __init__(self, gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss
