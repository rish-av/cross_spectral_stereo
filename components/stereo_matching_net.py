import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_
from components.components_cycle_gan import get_scheduler
from components.utils import *
import os
from components.utils import *
import math
from matplotlib import cm


class DispNet(nn.Module):

    def __init__(self, in_shape, in_channels, out_channels, out_scale):
        super(DispNet, self).__init__()
        shape = [(in_shape[0] // (2 ** i), in_shape[1] // (2 ** i)) for i in range(8)]
        nf = [16 * (2 ** x) for x in range(6)]
        self.e1 = Encoder(shape[0], shape[1], in_channels, nf[1], 7)
        self.e2 = Encoder(shape[1], shape[2], nf[1], nf[2], 5)
        self.e3 = Encoder(shape[2], shape[3], nf[2], nf[3])
        self.e4 = Encoder(shape[3], shape[4], nf[3], nf[4])
        self.e5 = Encoder(shape[4], shape[5], nf[4], nf[5])
        self.e6 = Encoder(shape[5], shape[6], nf[5], nf[5])
        self.e7 = Encoder(shape[6], shape[7], nf[5], nf[5])
        self.d7 = Decoder(shape[6], nf[5], nf[5], out_channels, nf[5])
        self.d6 = Decoder(shape[5], nf[5], nf[5], out_channels, nf[5])
        self.d5 = Decoder(shape[4], nf[5], nf[4], out_channels, nf[4])
        self.d4 = Decoder(shape[3], nf[4], nf[3], out_channels, nf[3], out_scale)
        self.d3 = Decoder(shape[2], nf[3], nf[2], out_channels, nf[2] + out_channels, out_scale)
        self.d2 = Decoder(shape[1], nf[2], nf[1], out_channels, nf[1] + out_channels, out_scale)
        self.d1 = Decoder(shape[0], nf[1], nf[0], out_channels, out_channels, out_scale)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        d7 = self.d7(e7, e6)
        d6 = self.d6(d7, e5)
        d5 = self.d5(d6, e4)
        d4, out4 = self.d4(d5, e3)
        out4up = F.interpolate(out4, e2.size()[2:], mode='bilinear')
        d3, out3 = self.d3(d4, torch.cat((e2, out4up), 1))
        out3up = F.interpolate(out3, e1.size()[2:], mode='bilinear')
        d2, out2 = self.d2(d3, torch.cat((e1, out3up), 1))
        out2up = F.interpolate(out2, x.size()[2:], mode='bilinear')
        d1, out1 = self.d1(d2, out2up)
        outs = [out1, out2, out3, out4]
        return outs


class Encoder(nn.Module):

    def __init__(self, in_shape, out_shape, in_channels, out_channels, ksize=3):
        super(Encoder, self).__init__()
        self.conv = Conv2dAP(in_shape, out_shape, in_channels, out_channels, ksize, 2, False)
        self.convbn = nn.BatchNorm2d(out_channels)
        self.convb = Conv2dAP(out_shape, out_shape, out_channels, out_channels, ksize, 1, False)
        self.convbbn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        conv = F.elu(self.convbn(self.conv(x)))
        convb = F.elu(self.convbbn(self.convb(conv)))
        return convb


class Decoder(nn.Module):

    def __init__(self, out_shape, in_channels, out_channels, final_out_channels, skip_channels, out_scale=0, ksize=3):
        super(Decoder, self).__init__()
        self.out_shape = out_shape
        self.conv = Conv2dAP(out_shape, out_shape, in_channels, out_channels, ksize, 1, False)
        self.convbn = nn.BatchNorm2d(out_channels)
        self.iconv = Conv2dAP(out_shape, out_shape, out_channels + skip_channels, out_channels, ksize, 1, False)
        self.iconvbn = nn.BatchNorm2d(out_channels)
        if out_scale > 0:
            self.out = Conv2dAP(out_shape, out_shape, out_channels, final_out_channels, ksize, 1, True)
            self.out_scale = out_scale
        else:
            self.out = None

    def forward(self, x, skip):
        upconv = F.elu(self.convbn(self.conv(F.interpolate(x, self.out_shape, mode='bilinear'))))
        iconv = F.elu(self.iconvbn(self.iconv(torch.cat((upconv, skip), 1))))
        if self.out is not None:
            out = self.out_scale * F.elu(self.out(iconv))
            return iconv, out
        else:
            return iconv


class Conv2dAP(nn.Module):
    def __init__(self, in_shape, out_shape, in_c, out_c, ksize, stride,
                 bias):
        super(Conv2dAP, self).__init__()
        if isinstance(ksize, int):
            ksize = (ksize, ksize)
        if isinstance(stride, int):
            stride = (stride, stride)
        pad = [0, 0]
        pad[0] = int(
            math.ceil((stride[0] * (out_shape[0] - 1) - in_shape[0] + ksize[0])
                      / 2))
        pad[1] = int(
            math.ceil((stride[1] * (out_shape[1] - 1) - in_shape[1] + ksize[1])
                      / 2))
        assert(pad[0] >= 0 and pad[1] >= 0)
        assert(pad[0] < ksize[0] and pad[1] < ksize[1])
        pad = tuple(pad)
        self.layer = nn.Conv2d(in_c, out_c, ksize, stride, pad, bias=bias)
        self.in_s = in_shape
        self.out_s = out_shape
        self.in_c = in_c
        self.out_c = out_c
        self.ksize = ksize
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        xs = x.size()
        assert(xs[1] == self.in_c and
               xs[2] == self.in_s[0] and xs[3] == self.in_s[1])
        y = self.layer(x)
        ys = y.size()
        assert(ys[2] == self.out_s[0] and ys[3] == self.out_s[1])
        return y
    

class StereoMatchingNet(nn.Module):

    def __init__(self,config,summarywriter= None):
        super(StereoMatchingNet,self).__init__()

        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu:0'

        self.inshape = config.crop_size
        self.netsmn = DispNet(self.inshape, 12, 2, 0.008).to(self.device)
        self.config = config
        self.summarywriter = summarywriter
        self.model_names = ["smn"]
        

    def setup(self,optimizer):
        self.optimizer = optimizer
        self.schedulers = get_scheduler(self.optimizer, None)

    def set_input(self,data):
        self.fake_A = data["fake_A"].to(self.device)
        self.fake_B = data["fake_B"].to(self.device)

        self.real_A = data["A"].to(self.device)
        self.real_B = data["B"].to(self.device)

    def forward(self):
        A_cat = torch.cat([self.real_A,self.fake_B],axis=1)
        B_cat = torch.cat([self.real_B,self.fake_A],axis=1)
        disps = self.netsmn(torch.cat((A_cat, B_cat), 1))
        ldisps = [disps[0][:, :1, :, :], disps[1][:, :1, :, :], disps[2][:, :1, :, :], disps[3][:, :1, :, :]]
        rdisps = [disps[0][:, 1:, :, :], disps[1][:, 1:, :, :], disps[2][:, 1:, :, :], disps[3][:, 1:, :, :]]
        self.ldisps = ldisps
        self.rdisps = rdisps

    def save_networks(self, epoch):
        """Save all the networks to the disk.
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.config.weights_dir, save_filename)
                net = getattr(self, 'net' + name)
                torch.save(net.state_dict(), save_path)

    def load_ckpts(self,epoch):
        if self.config.stm_pretrained:
            stm_path  = os.path.join(self.config.stm_pretrained,str(epoch) + "_net_smn.pth")
            self.netsmn.load_state_dict(torch.load(stm_path))


    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def get_loss(self):
        ldisps = self.ldisps
        rdisps = self.rdisps

        wrdisps = warp_pyramid(ldisps,rdisps,1)
        wldisps = warp_pyramid(rdisps,ldisps,-1)

        A_cat = torch.cat([self.real_A, self.fake_B],dim=1)
        B_cat = torch.cat([self.real_B, self.fake_A],dim=1)

        A_cats_pyramid = pyramid(A_cat)
        B_cats_pyramid = pyramid(B_cat)

        A_cats_warped_pyramid = warp_pyramid(A_cats_pyramid,rdisps,1)
        B_cats_warped_pyramid = warp_pyramid(B_cats_pyramid,ldisps,-1)

        smn_losses_total = 0
        for i,loss_weight in zip(range(4),self.config.multiscale_disp_weights):
            l_consist = l1_loss(ldisps[i], wrdisps[i])
            r_consist = l1_loss(rdisps[i], wldisps[i])

            l_l1 = l1_loss(B_cats_warped_pyramid[i], A_cats_pyramid[i])
            r_l1 = l1_loss(B_cats_pyramid[i], A_cats_warped_pyramid[i])


            l_ssim = dssim(B_cats_warped_pyramid[i], A_cats_pyramid[i])
            r_ssim = dssim(B_cats_pyramid[i], A_cats_warped_pyramid[i])

            l_photo = (1 - self.config.alpha_ap) * l_l1 + self.config.alpha_ap * l_ssim
            r_photo = (1 - self.config.alpha_ap) * r_l1 + self.config.alpha_ap * r_ssim
            ldisp_gradx, ldisp_grady = grad(ldisps[i])
            rdisp_gradx, rdisp_grady = grad(rdisps[i])
            rgb_gradx, rgb_grady = sobel(A_cats_pyramid[i])
            nir_gradx, nir_grady = sobel(B_cats_pyramid[i])
            l_easmooth = torch.exp(-l1_mean(rgb_gradx)) * ldisp_gradx.abs() + \
                torch.exp(-l1_mean(rgb_grady)) * ldisp_grady.abs()
            r_easmooth = torch.exp(-l1_mean(nir_gradx.cuda())) * rdisp_gradx.cuda().abs() + \
                torch.exp(-l1_mean(nir_grady.cuda())) * rdisp_grady.cuda().abs()
            smn_loss = self.config.alpha_lr * (l_consist.mean() + r_consist.mean()) + (l_photo.mean() + r_photo.mean()) + self.config.alpha_ds * (l_easmooth.mean() + r_easmooth.mean())
            smn_losses_total += loss_weight*smn_loss

        self.smn_loss = smn_losses_total
        return smn_losses_total
        

    def log_metrics(self,step):
        if self.summarywriter!= None:
            if self.config.stereo:
                self.summarywriter.add_scalar('smn_loss',self.smn_loss,step)


    def get_disparity_image(self,disp,maxdisp=-1,mask=None):

        disp = disp[0][0].detach().cpu().numpy()
        maxd = maxdisp
        if maxd < 0:
            maxd = np.max(disp)
        vals = disp/maxd
        img = cm.jet(vals)
        img[vals > 1] = [1,0,0,1]
        if mask is not None:
            img[mask != 1] = [0,0,0,1]
        img = img[:,:,0:3]
        return np.uint8(img*255)

    def optimize_parameters(self):
        self.forward()
        loss = self.get_loss()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    
    def tensor2im(self,tensor):
        tensor = tensor.permute(1,2,0).detach().cpu().numpy()*0.5 + 0.5
        return tensor

    def get_visuals(self):

        disparity_left = self.get_disparity_image(self.ldisps[0])
        disparity_right = self.get_disparity_image(self.rdisps[0])
        imgs_A = self.tensor2im(self.real_A[0])
        imgs_B = self.tensor2im(self.real_B[0])

        return disparity_left, disparity_right, imgs_A*255, imgs_B*255