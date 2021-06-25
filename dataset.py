import torch.nn as nn
import torch.utils.data as data
import cv2
import numpy as np
import torch
import os

class RgbNirData(data.Dataset):
    def __init__(self,config):
        self.basepath = config.basepath
        frame_seq = os.listdir(config.basepath)
        rgb_total = []
        nir_total = []


        resize = config.resize
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

    def __getitem__(self,index):
        rgb_img = cv2.imread(self.rgb_imgs[index])
        nir_img = cv2.imread(self.nir_imgs[index])

        rgb_img = cv2.resize(rgb_img,self.resize)
        nir_img = cv2.resize(nir_img,self.resize)

        ##you can have separate normalization
        rgb_img = (rgb_img - np.mean(rgb_img,axis=(0,1)))/(np.std(rgb_img,axis=(0,1)))
        nir_img = (nir_img - np.mean(nir_img))/np.std(nir_img)

        rgb_img = rgb_img.transpose(2,0,1)
        nir_img = nir_img.transpose(2,0,1)

        return np.float32(rgb_img[:,0:420,0:580]), np.float32(nir_img[:,0:420,0:580])

    def __len__(self):
        return len(self.rgb_imgs)