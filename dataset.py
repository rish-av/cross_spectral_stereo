import torch.nn as nn
import torch.utils.data as data
import cv2
import numpy as np
import torch
import os
import preprocessing_utils

class pittburgh_rgb_nir(data.Dataset):
    def __init__(self,config):
        self.basepath = config.basepath
        self.resize_factor = config.resize
        self.crop = config.crop_size

        frame_seq = os.listdir(config.basepath)
        rgb_total = []
        nir_total = []

        for frame in frame_seq:
            frame_path = os.path.join(self.basepath,frame)
            rgb_frame_list = os.listdir(os.path.join(frame_path,"RGBResize"))
            nir_frame_list = os.listdir(os.path.join(frame_path,"NIRResize"))

            rgb_full_paths = [os.path.join(os.path.join(frame_path,"RGBResize"),rgb_image) for rgb_image in rgb_frame_list]
            nir_full_paths = [os.path.join(os.path.join(frame_path,"NIRResize"),nir_image) for nir_image in nir_frame_list]

            rgb_total+=rgb_full_paths
            nir_total+=nir_full_paths

        self.rgb_imgs = sorted(rgb_total)
        self.nir_imgs = sorted(nir_total)

        assert(len(self.rgb_imgs)==len(self.nir_imgs))

    def __getitem__(self,index):
        rgb_img = cv2.imread(self.rgb_imgs[index])
        nir_img = cv2.imread(self.nir_imgs[index])

        h,w,_ = rgb_img.shape
        new_size = (w//self.resize_factor,h//self.resize_factor)

        rgb_img = cv2.resize(rgb_img,new_size)
        nir_img = cv2.resize(nir_img,new_size)


        rgb_nir_cat = np.concatenate([rgb_img,nir_img],axis=2)
        rgb_nir_cat = preprocessing_utils._get_random_crop(rgb_nir_cat,self.crop[0],self.crop[1])

        rgb_img = rgb_nir_cat[:,:,:3]
        nir_img = rgb_nir_cat[:,:,3:]

        ##you can have different normalization

        org_A = rgb_img.transpose(2,0,1)
        org_B = nir_img.transpose(2,0,1)

        rgb_img = (rgb_img - np.mean(rgb_img,axis=(0,1)))/(np.std(rgb_img,axis=(0,1)))
        nir_img = (nir_img - np.mean(nir_img))/np.std(nir_img)


        rgb_img = rgb_img.transpose(2,0,1)
        nir_img = nir_img.transpose(2,0,1)


        return {"real_A":np.float32(rgb_img), "real_B":np.float32(nir_img),
                "org_A":np.float32(org_A),"org_B":np.float32(org_B)}

    def __len__(self):
        return len(self.rgb_imgs)
