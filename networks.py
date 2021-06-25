import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)  # add skip connections
        return out

class Generator(nn.Module):
    def __init__(self,n_downsampling,ngf=64,norm_layer=nn.BatchNorm2d,use_bias=True):
        super(Generator,self).__init__()
        G = []
        for i in range(n_downsampling): 
            mult = 2 ** (n_downsampling - i)
            G += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        G += [nn.ReflectionPad2d(3)]

        G+= [nn.Conv2d(ngf, 3, kernel_size=7, padding=0)]
        G += [nn.Tanh()]
        self.G = nn.Sequential(*G)

    def forward(self,x):
        return self.G(x)


class FeatureExtraction(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(FeatureExtraction, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):     

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        self.F = nn.Sequential(*model)

    def forward(self, input):
        F = self.F(input)
        return F

class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


class DispNetS(nn.Module):

    def __init__(self, alpha=10, beta=0.01):
        super(DispNetS, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.training = False

        conv_planes = [32, 64, 128, 256, 512, 512, 512]
        self.conv1 = self.downsample_conv(3, conv_planes[0], 7)
        self.conv2 = self.downsample_conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = self.downsample_conv(conv_planes[1], conv_planes[2])
        self.conv4 = self.downsample_conv(conv_planes[2], conv_planes[3])
        self.conv5 = self.downsample_conv(conv_planes[3], conv_planes[4])
        self.conv6 = self.downsample_conv(conv_planes[4], conv_planes[5])
        self.conv7 = self.downsample_conv(conv_planes[5], conv_planes[6])

        upconv_planes = [512, 512, 256, 128, 64, 32, 16]
        self.upconv7 = self.upconv(conv_planes[6],   upconv_planes[0])
        self.upconv6 = self.upconv(upconv_planes[0], upconv_planes[1])
        self.upconv5 = self.upconv(upconv_planes[1], upconv_planes[2])
        self.upconv4 = self.upconv(upconv_planes[2], upconv_planes[3])
        self.upconv3 = self.upconv(upconv_planes[3], upconv_planes[4])
        self.upconv2 = self.upconv(upconv_planes[4], upconv_planes[5])
        self.upconv1 = self.upconv(upconv_planes[5], upconv_planes[6])

        self.iconv7 = self.conv(upconv_planes[0] + conv_planes[5], upconv_planes[0])
        self.iconv6 = self.conv(upconv_planes[1] + conv_planes[4], upconv_planes[1])
        self.iconv5 = self.conv(upconv_planes[2] + conv_planes[3], upconv_planes[2])
        self.iconv4 = self.conv(upconv_planes[3] + conv_planes[2], upconv_planes[3])
        self.iconv3 = self.conv(1 + upconv_planes[4] + conv_planes[1], upconv_planes[4])
        self.iconv2 = self.conv(1 + upconv_planes[5] + conv_planes[0], upconv_planes[5])
        self.iconv1 = self.conv(1 + upconv_planes[6], upconv_planes[6])

        self.predict_disp4 = self.predict_disp(upconv_planes[3])
        self.predict_disp3 = self.predict_disp(upconv_planes[4])
        self.predict_disp2 = self.predict_disp(upconv_planes[5])
        self.predict_disp1 = self.predict_disp(upconv_planes[6])



    def downsample_conv(self,in_planes, out_planes, kernel_size=3):
        return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True)
        )

    def predict_disp(self,in_planes):
        return nn.Sequential(
        nn.Conv2d(in_planes, 1, kernel_size=3, padding=1),
        nn.Sigmoid()
        )


    def conv(self,in_planes, out_planes):
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )


    def upconv(self,in_planes, out_planes):
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )


    def crop_like(self,input, ref):
        assert(input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
        return input[:, :, :ref.size(2), :ref.size(3)]

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        out_upconv7 = self.crop_like(self.upconv7(out_conv7), out_conv6)
        concat7 = torch.cat((out_upconv7, out_conv6), 1)
        out_iconv7 = self.iconv7(concat7)

        out_upconv6 = self.crop_like(self.upconv6(out_iconv7), out_conv5)
        concat6 = torch.cat((out_upconv6, out_conv5), 1)
        out_iconv6 = self.iconv6(concat6)

        out_upconv5 = self.crop_like(self.upconv5(out_iconv6), out_conv4)
        concat5 = torch.cat((out_upconv5, out_conv4), 1)
        out_iconv5 = self.iconv5(concat5)

        out_upconv4 = self.crop_like(self.upconv4(out_iconv5), out_conv3)
        concat4 = torch.cat((out_upconv4, out_conv3), 1)
        out_iconv4 = self.iconv4(concat4)
        disp4 = self.alpha * self.predict_disp4(out_iconv4) + self.beta

        out_upconv3 = self.crop_like(self.upconv3(out_iconv4), out_conv2)
        disp4_up = self.crop_like(F.interpolate(disp4, scale_factor=2, mode='bilinear', align_corners=False), out_conv2)
        concat3 = torch.cat((out_upconv3, out_conv2, disp4_up), 1)
        out_iconv3 = self.iconv3(concat3)
        disp3 = self.alpha * self.predict_disp3(out_iconv3) + self.beta

        out_upconv2 = self.crop_like(self.upconv2(out_iconv3), out_conv1)
        disp3_up = self.crop_like(F.interpolate(disp3, scale_factor=2, mode='bilinear', align_corners=False), out_conv1)
        concat2 = torch.cat((out_upconv2, out_conv1, disp3_up), 1)
        out_iconv2 = self.iconv2(concat2)
        disp2 = self.alpha * self.predict_disp2(out_iconv2) + self.beta

        out_upconv1 = self.crop_like(self.upconv1(out_iconv2), x)
        disp2_up = self.crop_like(F.interpolate(disp2, scale_factor=2, mode='bilinear', align_corners=False), x)
        concat1 = torch.cat((out_upconv1, disp2_up), 1)
        out_iconv1 = self.iconv1(concat1)
        disp1 = self.alpha * self.predict_disp1(out_iconv1) + self.beta

        if self.training:
            return disp1, disp2, disp3, disp4
        else:
            return disp1




        


