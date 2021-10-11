import torch.nn as nn
import torch.utils.data as data
import cv2
import numpy as np
import torch
import os
from torchvision import transforms
from pathlib import Path


def _get_random_crop(image, crop_height, crop_width):
	x, y =0, 0

	max_x = image.shape[1] - crop_width
	max_y = image.shape[0] - crop_height
	if max_x == 0 and max_y!=0:
		x = 0
		y = np.random.randint(0, max_y)

	if max_y == 0 and max_x!=0:
		y = 0
		x = np.random.randint(0, max_x)

	if max_x!=0 and max_y!=0:
		x = np.random.randint(0, max_x)
		y = np.random.randint(0, max_y)

	crop = image[y: y + crop_height, x: x + crop_width]

	return crop

class pittburgh_rgb_nir(data.Dataset):
    def __init__(self,config,mode='train'):
        self.basepath = './data/'
        self.resize_factor = 1
        self.crop = config.crop_size
        self.mode = mode

        frame_seq_train = os.listdir(self.basepath)
        frame_seq_test = ["20170222_0951","20170222_1423","20170223_1639","20170224_0742"]

        rgb_total = []
        nir_total = []

        for frame in frame_seq_train:
            frame_path = os.path.join(self.basepath,frame)
            rgb_frame_list = os.listdir(os.path.join(frame_path,"RGBResize"))
            nir_frame_list = os.listdir(os.path.join(frame_path,"NIRResize"))

            rgb_full_paths = [os.path.join(os.path.join(frame_path,"RGBResize"),rgb_image) for rgb_image in rgb_frame_list]
            nir_full_paths = [os.path.join(os.path.join(frame_path,"NIRResize"),nir_image) for nir_image in nir_frame_list]

            rgb_total+=rgb_full_paths
            nir_total+=nir_full_paths

        rgb_total_test = []
        nir_total_test = []

        for frame in frame_seq_test:
            frame_path = os.path.join(self.basepath,frame)
            rgb_frame_list = os.listdir(os.path.join(frame_path,"RGBResize"))
            nir_frame_list = os.listdir(os.path.join(frame_path,"NIRResize"))

            rgb_full_paths = [os.path.join(os.path.join(frame_path,"RGBResize"),rgb_image) for rgb_image in rgb_frame_list]
            nir_full_paths = [os.path.join(os.path.join(frame_path,"NIRResize"),nir_image) for nir_image in nir_frame_list]

            rgb_total_test+=rgb_full_paths
            nir_total_test+=nir_full_paths

        self.rgb_imgs = sorted(rgb_total)
        self.nir_imgs = sorted(nir_total)

        self.rgb_imgs_test = sorted(rgb_total_test)
        self.nir_imgs_test = sorted(nir_total_test)

        assert(len(self.rgb_imgs)==len(self.nir_imgs))

    def __getitem__(self,index):
        if self.mode == 'train':
            rgb_list, nir_list = self.rgb_imgs, self.nir_imgs
        else:
            rgb_list, nir_list = self.rgb_imgs_test, self.nir_imgs_test


        rgb_img = cv2.imread(rgb_list[index])
        nir_img = cv2.imread(nir_list[index])

        h,w,_ = rgb_img.shape
        new_size = (w//self.resize_factor,h//self.resize_factor)

        rgb_img = cv2.resize(rgb_img,new_size)
        nir_img = cv2.resize(nir_img,new_size)


        rgb_nir_cat = np.concatenate([rgb_img,nir_img],axis=2)
        rgb_nir_cat = _get_random_crop(rgb_nir_cat,self.crop[0],self.crop[1])

        rgb_img = rgb_nir_cat[:,:,:3]
        nir_img = rgb_nir_cat[:,:,3:]

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(std=(0.5,0.5,0.5), mean = (0.5,0.5,0.5))
        ])

        rgb_img = transform(rgb_img)
        nir_img = transform(nir_img)

        return {"A":rgb_img, "B":nir_img, "A_paths":rgb_list[index],"B_paths":nir_list[index]}

    def __len__(self):
        if self.mode == 'train':
            return len(self.rgb_imgs)
        else:
            return len(self.rgb_imgs_test)


