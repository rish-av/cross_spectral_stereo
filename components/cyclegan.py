import torch
import itertools
import torch
from components.components_cycle_gan import BaseModel, GANLoss, define_D, define_G
import random
import os
import torch.nn as nn

class ImagePool():
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images



class CycleGANModel(BaseModel):

    def __init__(self,config,summarywriter= None):
        """Initialize the CycleGAN class.
        Parameters:
            config (config file)
        """
        BaseModel.__init__(self)

        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu:0'
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']

        self.summarywriter = summarywriter

        self.config = config
        if self.isTrain and False > 0.0:
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else: 
            self.model_names = ['G_A', 'G_B']
        self.netG_A = define_G(3, 3, 64, 'resnet_9blocks', 'instance',
                                        False, 'normal').to(self.device)
        self.netG_B = define_G(3, 3, 64, 'resnet_9blocks', 'instance',
                                        False, 'normal').to(self.device)

        if self.isTrain:  # define discriminators
            self.netD_A = define_D().to(self.device)
            self.netD_B = define_D().to(self.device)

        if self.isTrain:
            self.fake_A_pool = ImagePool(50)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(50)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = GANLoss('lsgan').to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=0.0002, betas=(0.99, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=0.0002, betas=(0.99, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = True
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']


    def load_ckpts(self,epoch):
        if self.config.gan_pretrained:
            net_GA_path = os.path.join(self.config.gan_pretrained, str(epoch) + '_' + 'net_G_A.pth')
            net_GB_path = os.path.join(self.config.gan_pretrained, str(epoch) + '_' + 'net_G_B.pth')
            net_DA_path = os.path.join(self.config.gan_pretrained, str(epoch) + '_' + 'net_D_A.pth')
            net_DB_path = os.path.join(self.config.gan_pretrained, str(epoch) + '_' + 'net_D_B.pth')

            self.netG_A.load_state_dict(torch.load(net_GA_path))
            self.netG_B.load_state_dict(torch.load(net_GB_path))
            self.netD_A.load_state_dict(torch.load(net_DA_path))
            self.netD_B.load_state_dict(torch.load(net_DB_path))
        else:
            print('Training GAN from scratch...')

    def save_networks(self, epoch):
        """Save all the networks to the disk.
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.config.weights_dir, save_filename)
                net = getattr(self, 'net' + name)
                torch.save(net.state_dict(), save_path)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D


    def optimize_auxilary(self,aux_loss):
        self.optimizer_G.zero_grad()
        self.aux_loss = self.config.alpha_aux * aux_loss
        self.aux_loss.backward()
        self.optimizer_G.step()

    def log_metrics(self,step):

        if self.summarywriter!= None:
            if self.config.warmup:
                self.summarywriter.add_scalar('GA_loss',self.loss_G_A,step)
                self.summarywriter.add_scalar('GB_loss',self.loss_G_B,step)
                self.summarywriter.add_scalar('DA_loss',self.loss_D_A,step)
                self.summarywriter.add_scalar('DB_loss',self.loss_D_B,step)
                
            elif self.config.auxilary:
                self.summarywriter.add_scalar('aux_loss',self.aux_loss,step)

    

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.config.lambda_r
        lambda_A = self.config.lambda_c
        lambda_B = self.config.lambda_c
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B)  * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A)  * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

    def get_images(self):

        return self.fake_B, self.fake_A, self.real_A, self.real_B


    def tensor2im(self,tensor):
        tensor = tensor.permute(1,2,0).detach().cpu().numpy()*0.5 + 0.5
        return tensor

    def get_visuals(self):
        visuals = {}
        visuals["real_A"] = self.tensor2im(self.real_A[0])*255.
        visuals["real_B"] = self.tensor2im(self.real_B[0])*255.
        visuals["fake_A"] = self.tensor2im(self.fake_A[0])*255.
        visuals["fake_B"] = self.tensor2im(self.fake_B[0])*255.

        return visuals