import time
from tqdm import tqdm
from numpy.lib.type_check import real
from dataset import pittburgh_rgb_nir
import yaml
import argparse
from attrdict import AttrDict
from components.cyclegan import CycleGANModel
from components.stereo_matching_net import StereoMatchingNet
from components.utils import get_summary_writer, pyramid, warp_pyramid
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as Func
import torch.nn as nn


parser = argparse.ArgumentParser()
parser.add_argument('--config_file',default='./configs/pittsburgh.yaml',help='path to the config file')
args = parser.parse_args()
l2_loss = nn.MSELoss()


with open(args.config_file) as fp:
    config = yaml.safe_load(fp)
    config = AttrDict(config)


def get_auxilary_loss(real_A, real_B, fake_A, fake_B, ldisps, rdisps):
    warped_As = pyramid(real_A)
    warped_Bs = pyramid(real_B)

    fake_As = pyramid(fake_A)
    fake_Bs = pyramid(fake_B)

    warped_As = warp_pyramid(warped_As,rdisps,1)
    warped_Bs = warp_pyramid(warped_Bs,ldisps,-1)
    
    net_loss = 0.
    for warped_A, warped_B, fake_A_s, fake_B_s,wt in zip(warped_As, warped_Bs, fake_As, fake_Bs, config.multiscale_disp_weights):
        scale_loss = l2_loss(warped_A,fake_A_s) + l2_loss(warped_B,fake_B_s)
        net_loss+=wt*scale_loss

    return net_loss



def device():
    if torch.cuda.is_available():
        return 'cuda:0'
    else:
        return 'cpu:0'

if __name__ == '__main__':
    datas = pittburgh_rgb_nir(config)
    dataset_size = len(datas)
    dataset = torch.utils.data.DataLoader(datas,config.batch_size,shuffle=True)

    warmup = config.warmup
    stereo = config.stereo
    auxilary = config.auxilary
    epochs = config.epochs

    summarywriter = get_summary_writer(rootdir=config.summary_root)

    spectral_net = CycleGANModel(config)
    stereo_matching_net = StereoMatchingNet(config)

    spectral_net.load_ckpts(epoch=config.pretrained_epoch)
    stereo_matching_net.load_ckpts(epoch=config.pretrained_epoch)
    
    
    optim_smn = torch.optim.Adam(stereo_matching_net.parameters(), lr=0.0002, betas=(0.99, 0.999))

    spectral_net.setup()
    stereo_matching_net.setup(optim_smn)


    total_iters = 0            
    epoch_start_time = time.time()
    for epoch in range(0, epochs):
        epoch_iter = 0
        spectral_net.update_learning_rate()
        for i, data in tqdm(enumerate(dataset)):  
            total_iters += 1
            epoch_iter += 1

            step = epoch*len(dataset) + i
            
            #for warmup epochs where only GAN is trained for spectral translation step 1 and 2
            if warmup:
                stereo_matching_net.set_requires_grad(stereo_matching_net,False)
                spectral_net.set_input(data)         
                spectral_net.optimize_parameters()
                spectral_net.log_metrics(step=step)

                if total_iters % config.weights_freq == 0:   
                    save_suffix = 'latest'
                    spectral_net.save_networks(save_suffix)


            #all steps after warmp epochs (all steps --> 1,2,3,4)
            else:

                #step 1,2
                stereo_matching_net.set_requires_grad(stereo_matching_net,False)
                spectral_net.set_input(data)         
                spectral_net.optimize_parameters()
                spectral_net.log_metrics(step=step)

                #step 3
                spectral_net.set_requires_grad([spectral_net.netG_A,spectral_net.netG_B, spectral_net.netD_B, spectral_net.netD_A], False)
                fake_B,fake_A,_,_ = spectral_net.get_images()
                data["fake_A"] = fake_A.detach()
                data["fake_B"] = fake_B.detach()

                stereo_matching_net.set_requires_grad(stereo_matching_net,True)
                stereo_matching_net.set_input(data)
                stereo_matching_net.optimize_parameters()
                stereo_matching_net.log_metrics(step=step)


                #step 4
                stereo_matching_net.set_requires_grad(stereo_matching_net,False)
                spectral_net.set_requires_grad([spectral_net.netG_A,spectral_net.netG_B, spectral_net.netD_B, spectral_net.netD_A],True)
                spectral_net.set_input(data)
                spectral_net.forward()
                fake_B,fake_A,_,_ = spectral_net.get_images()

                data["fake_A"] = fake_A
                data["fake_B"] = fake_B

                stereo_matching_net.set_input(data)
                stereo_matching_net.forward()
                ldisps, rdisps = stereo_matching_net.ldisps, stereo_matching_net.rdisps

                aux_loss = get_auxilary_loss(data["A"].to(device()), data["B"].to(device()), fake_A.to(device()), 
                fake_B.to(device()), ldisps, rdisps)
                spectral_net.optimize_auxilary(aux_loss)
                spectral_net.log_metrics(step=step)

                if total_iters % config.weights_freq == 0:   
                    save_suffix = 'latest'
                    spectral_net.save_networks(save_suffix)
                    stereo_matching_net.save_networks(save_suffix)

        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        spectral_net.save_networks(epoch)
        stereo_matching_net.save_networks(epoch)
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, 20, time.time() - epoch_start_time))