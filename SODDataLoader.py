#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loading Dataset for SOD
@author:Yan Runming
"""
from __future__ import print_function, division
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset
import os

class RescaleT(object):

    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        self.output_size = output_size

    def __call__(self,sample):
        image = sample['image']
        # resize the image and convert image from range [0,255] to [0,1]
        img = transform.resize(image,(self.output_size,self.output_size),mode='constant')
        return {'image':img}


class ToTensor(object):

    def __call__(self, sample):
        image = sample['image']

        tmpImg = np.zeros((image.shape[0],image.shape[1],3))

        image = image/np.max(image)

        if image.shape[2]==1:
            tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
        else:
            tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
            tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

        # change the r,g,b to b,r,g from [0,255] to [0,1]
        tmpImg = tmpImg.transpose((2, 0, 1))

        return {'image': torch.from_numpy(tmpImg)}


class SalObjDataset(Dataset):
    def __init__(self,img_name_list,image_dir,transform=None):
        self.image_dir=image_dir
        self.image_name_list = img_name_list
        self.transform = transform

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self,idx):

        image = io.imread(os.path.join(self.image_dir,self.image_name_list[idx]))
        if(2==len(image.shape)):
            image = image[:,:,np.newaxis]

        sample = {'image':image}
        if self.transform:
            sample = self.transform(sample)

        return sample

