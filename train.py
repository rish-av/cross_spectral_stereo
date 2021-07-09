from torch.nn.modules.loss import L1Loss
from losses import *
from networks import FeatureExtraction,DispNetS,PixelDiscriminator, Generator
from dataset import *
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm
from torch.nn import DataParallel as DP
import torch.nn as nn

import baseutils
import torch.nn.functional as Func

import yaml
import argparse
from tqdm import tqdm
import logging

from os.path import join

parser = argparse.ArgumentParser()
parser.add_argument('--config_file',default='./configs/pittsburgh.yaml',help='path to the config file')

args = parser.parse_args()

with open(args.config_file) as fp:
    config = yaml.safe_load(fp)
    config = baseutils.AttrDict(config)


if config:
    logging.info("Config %s Loaded!", args.config_file)


device = baseutils.get_device()

F = DP(FeatureExtraction(3,None).to(device),config.device_ids)
GA = DP(Generator(2).to(device),config.device_ids)
GB = DP(Generator(2).to(device),config.device_ids)
STM = DP(DispNetS(training=True).to(device),config.device_ids)
DA = DP(PixelDiscriminator(3).to(device),config.device_ids)
DB = DP(PixelDiscriminator(3).to(device),config.device_ids) 



opt_gen = torch.optim.Adam(list(GA.parameters()) + list(GB.parameters()) +list(F.parameters()),lr=0.0002)
opt_disc = torch.optim.Adam(list(DA.parameters()) + list(DB.parameters()),lr=0.0002)
opt_stm = torch.optim.Adam(STM.parameters(),lr=0.0002)

gan_loss = GANLoss()
cycle_loss = nn.MSELoss()
identity_loss = nn.L1Loss()


def load_weights(path):

    state_dict = torch.load(path)
    epoch = state_dict['epoch']
    GA.load_state_dict(state_dict['GA'])
    GB.load_state_dict(state_dict['GB'])
    DA.load_state_dict(state_dict['DA'])
    DB.load_state_dict(state_dict['DB'])
    F.load_state_dict(state_dict['F'])
    STM.load_state_dict(state_dict['STM'])


    return epoch


def save_weights(epoch):

    save_dict = {"GA":GA.state_dict(),"GB":GB.state_dict(),"DA":DA.state_dict(),"DB":DB.state_dict(),
        "F":F.state_dict(),"STM":STM.state_dict(),"epoch":epoch}

    torch.save(save_dict,join(config.weights_dir,"matching_net_",epoch,"_",iteration,".pth"))

def get_dataloaders():

    dataset = pittburgh_rgb_nir(config)

    train_sampler, val_sampler = baseutils._split(dataset,config.val_percent)
    train_loader = torch.utils.data.DataLoader(dataset,config.batch_size,sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset,1,sampler=val_sampler)

    return train_loader, val_loader



def optimize_generators(real_A, real_B, fake_A, fake_B, rec_A, rec_B,train=True):
        lambda_idt = config.lambda_i
        lambda_c = config.lambda_c
        if lambda_idt > 0: 
            idt_A = GA(real_B)
            loss_idt_A = identity_loss(idt_A, real_B)  * lambda_idt
            idt_B = GB(real_A)
            loss_idt_B = identity_loss(idt_B, real_A)  * lambda_idt
        else:
            loss_idt_A = 0
            loss_idt_B = 0

        loss_G_A = gan_loss(DA(fake_B), True)
        loss_G_B = gan_loss(DB(fake_A), True)
        loss_cycle_A = cycle_loss(rec_A, real_A) * lambda_c
        loss_cycle_B = cycle_loss(rec_B, real_B) * lambda_c
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B

        if train:
            loss_G.backward()
        return loss_G



def optimize_discriminator(real,fake,D,train=True):

    pred_real = D(real)
    loss_D_real = gan_loss(pred_real, True)
    pred_fake = D(fake.detach())
    loss_D_fake = gan_loss(pred_fake, False)
    loss_D = (loss_D_real + loss_D_fake) * 0.5

    if train:
        loss_D.backward()
    return loss_D


 

def train():
    writer = baseutils.get_summary_writer(config.summary_root)

    warmup = config.warmup
    stereo = config.stereo
    auxilary = config.auxilary
    epochs = config.epochs


    train_loader, val_loader = get_dataloaders()
    train_size = len(train_loader)
    val_size = len(val_loader)
    num_devices = len(config.device_ids)


    if config.pretrained:
        start_epoch = load_weights(config.pretrained)
    else:
        start_epoch = 0

    for epoch in range(start_epoch,epochs):
        for i,data in tqdm(enumerate(train_loader),total=len(train_loader)):


            real_A = data['real_A'].to(device)
            real_B = data['real_B'].to(device)


            non_normalized_A = data['org_A']
            non_normalized_B = data['org_B']

            iteration = epoch*len(train_loader) + i

            if warmup:
                baseutils.block_grad(STM)
                fake_B = GA(F(real_A))
                fake_A = GB(F(real_B))
                rec_A = GB(F(fake_B))
                rec_B = GA(F(fake_A))

                #step_1
                baseutils.block_grad(GA)
                baseutils.block_grad(GB)
                baseutils.block_grad(F)

                opt_disc.zero_grad()
                loss_DA = optimize_discriminator(real_A,fake_A,DA)
                loss_DB = optimize_discriminator(real_B,fake_B,DB)
                opt_disc.step()


                #step_2
                baseutils.start_grad(GA)
                baseutils.start_grad(GB)
                baseutils.start_grad(F)

                baseutils.block_grad(DA)
                baseutils.block_grad(DB)

                opt_gen.zero_grad()
                loss_G = optimize_generators(real_A,real_B,fake_A,fake_B,rec_A,rec_B)
                opt_gen.step()

                baseutils.start_grad(DA)
                baseutils.start_grad(DB)


                scalars_to_write = {"Train/Da_loss":loss_DA,"Train/Db_loss":loss_DB, "Train/G_loss":loss_G}
                baseutils._log(writer,iteration,scalars=scalars_to_write)

            #step 3
            if stereo:
                # optimize STM

                baseutils.start_grad(STM)
                baseutils.block_grad(GA)
                baseutils.block_grad(GB)
                baseutils.block_grad(DA)
                baseutils.block_grad(DB)
                baseutils.block_grad(F)

                fake_B = GA(F(real_B))
                fake_A = GB(F(real_A))


                dispnet_inp_1 = torch.cat([real_A,fake_B],dim=1)
                dispnet_inp_2 = torch.cat([real_B,fake_A],dim=1)

                dispsl = STM(dispnet_inp_1)
                dispsr = STM(dispnet_inp_2)

                loss_stm = multi_scale_loss(config,real_A,real_B,dispsl,dispsr)

                opt_stm.zero_grad()
                loss_stm.backward()
                opt_stm.step()


                scalars_to_write = {"Train/loss_stm":loss_stm}
                baseutils._log(writer,step=iteration,scalars=scalars_to_write)

            if auxilary:
                # optimize based on auxilary loss
                baseutils.block_grad(STM)
                baseutils.block_grad(DA)
                baseutils.block_grad(DB)

                baseutils.start_grad(GA)
                baseutils.start_grad(GB)
                baseutils.start_grad(F)


                fake_B = GA(F(real_B))
                fake_A = GB(F(real_A))

                dispnet_inp_1 = torch.cat([real_A,fake_B],dim=1)
                dispnet_inp_2 = torch.cat([real_B,fake_A],dim=1)

                dispsl = STM(dispnet_inp_1)
                dispsr = STM(dispnet_inp_2)

                displ = dispsl[0]
                dispr = dispsr[0]


                aux_loss = config.alpha_aux*auxilary_loss(real_A,real_B,fake_A,fake_B,displ,displr)

                opt_gen.zero_grad()
                aux_loss.backward()
                opt_gen.step()

                scalars_to_write = {"Train/auxilary_loss":aux_loss}
                baseutils._log(writer,step=iteration,scalars=scalars_to_write)


            if iteration>0. and iteration % config.tensorboard_val_freq==0 :
                
                with torch.no_grad():
                    for j,data in tqdm(enumerate(val_loader),total=len(val_loader)):
                        iteration_val = j + len(val_loader)*epoch


                        real_A = data['real_A'].to(device)
                        real_B = data['real_B'].to(device)


                        non_normalized_A = data['org_A']
                        non_normalized_B = data['org_B']

                        fake_B = GA(F(real_B))
                        fake_A = GB(F(real_A))

                        dispnet_inp_1 = torch.cat([real_A,fake_B],dim=1)
                        dispnet_inp_2 = torch.cat([real_B,fake_A],dim=1)

                        dispsl = STM(dispnet_inp_1)
                        dispsr = STM(dispnet_inp_2)

                        displ = dispsl[0]
                        dispr = dispsr[0]

                        loss_G = optimize_generators(real_A,real_B,fake_A,fake_B,rec_A,rec_B,train=False)
                        loss_DA = optimize_discriminator(real_A,fake_A,DA,train=False)
                        loss_DB = optimize_discriminator(real_B,fake_B,DB,train=False)

                        loss_stm = multi_scale_loss(config,real_A,real_B,dispsl,dispsr)
                        aux_loss = config.alpha_aux*auxilary_loss(real_A,real_B,fake_A,fake_B,displ,dispr)


                        scalars_to_write = {"Val/Da_loss":loss_DA,"Val/Db_loss":loss_DB, 
                                        "Val/G_loss":loss_G,"Val/loss_stm":loss_stm,"Val/auxilary_loss":aux_loss}

                        baseutils._log(writer,iteration_val,scalars=scalars_to_write)

                        if iteration > 0. and iteration_val % config.image_on_tensorboard == 0:
                            images_to_write = {"A":non_normalized_A[0],"B":non_normalized_B[0],"displ":displ[0],
                                    "dispr":dispr[0],"fake_A":fake_A[0], "fake_B":fake_B[0]}
                            baseutils._log(writer,iteration_val,images=images_to_write)


            if iteration > 0. and iteration % config.weights_freq == 0:
                save_weights(epoch)





if __name__ == '__main__':
    train()
            









        
            
            


            
        

        



    

