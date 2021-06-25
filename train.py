from torch.nn.modules.loss import L1Loss
from losses import *
from utils import *
from networks import FeatureExtraction,DispNetS,PixelDiscriminator, Generator
from dataset import *
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm
from torch.nn import DataParallel as DP

import baseutils
import torch.nn.functional as F

import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config_file',default='pittsburgh_config.yaml',help='path to the config file')

args = parser.parse_args()

with open(args.config_file) as fp:
    config = yaml.safe_load(fp)

F = DP(FeatureExtraction(3,None).cuda(),config.device_ids)
GA = DP(Generator(2).cuda(),config.device_ids)
GB = DP(Generator(2).cuda(),config.device_ids)
STM = DP(DispNetS().cuda(),config.device_ids)
DA = DP(PixelDiscriminator(3).cuda(),config.device_ids)
DB = DP(PixelDiscriminator(3).cuda(),config.device_ids) 

alpha_ap = config.ap
alpha_ds = config.ds
alpha_lr = config.lr

opt_gen = torch.optim.Adam(list(GA.parameters()) + list(GB.parameters()) +list(F.parameters()),lr=0.0002)
opt_disc = torch.optim.Adam(list(DA.parameters()) + list(DB.parameters()),lr=0.0002)
opt_stm = torch.optim.Adam(STM.parameters(),lr=0.0002)


def optimize_generators(real_A, real_B, fake_A, fake_B, rec_A, rec_B):
        lambda_idt = config.lambda_identity
        lambda_A = config.lambda_A
        lambda_B = config.lambda_B
        if lambda_idt > 0: 
            idt_A = GA(real_B)
            loss_idt_A = identity_loss(idt_A, real_B) * lambda_B * lambda_idt
            idt_B = GB(real_A)
            loss_idt_B = identity_loss(idt_B, real_A) * lambda_A * lambda_idt
        else:
            loss_idt_A = 0
            loss_idt_B = 0

        loss_G_A = gan_loss(DA(fake_B), True)
        loss_G_B = gan_loss(DB(fake_A), True)
        loss_cycle_A = cycle_loss(rec_A, real_A) * lambda_A
        loss_cycle_B = cycle_loss(rec_B, real_B) * lambda_B
        loss_G = oss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
        loss_G.backward()

        return loss_G

def optimize_discriminator(real,fake,D):

    pred_real = D(real)
    loss_D_real = gan_loss(pred_real, True)
    pred_fake = D(fake.detach())
    loss_D_fake = gan_loss(pred_fake, False)
    loss_D = (loss_D_real + loss_D_fake) * 0.5
    loss_D.backward()

    return loss_D




 

def train(num_epocs=20):
    writer = baseutils.get_summary_writer(config.summary_root)
    warmup = config.warmup
    stereo = config.stereo
    auxilary = config.auxilary

    if warmup:
    
        fake_B = GA(real_B)
        fake_A = GB(real_A)


        rec_A = GB(fake_B)
        rec_B = GA(fake_A)

        baseutils.block_grad(DA)
        baseutils.block_grad(DB)

        opt_gen.zero_grad()
        optimize_generators(real_A,real_B,fake_A,fake_B,rec_A,rec_B)
        opt_gen.step()


        baseutils.start_grad(DA)
        baseutils.start_grad(DB)


        opt_disc.zero_grad()
        optimize_discriminator(real_A,fake_A,DA)
        optimize_discriminator(real_B,fake_B,DB)
        opt_disc.step()


    if stereo:

        # optimize STM

    if auxilary:
        # optimize based on auxilary loss




if __name__ == '__main__':
    train(False)
            









        
            
            


            
        

        



    

