# -*- coding: utf-8 -*-
"""
SOD Module
@author: Yan Runming
"""

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import cv2
import os

from SODDataLoader import RescaleT
from SODDataLoader import ToTensor
from SODDataLoader import SalObjDataset
from module.SODmodule import SODNet

model_dir = './pretrained/SOD_Param.pth'
SODNet = SODNet(3,1)

def SODdatasets(net, dataloader,predi_dir,img_name_list,image_dir):
    
    checkpoint = torch.load(model_dir, map_location='cpu')
    net.load_state_dict(checkpoint['model'])
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    
    i=0
    for i_test, data_test in enumerate(dataloader):
        
        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)
        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)
        d0 = net(inputs_test)
        pred = d0[:,0,:,:]
        # normalization
        maxd = torch.max(pred)
        mind = torch.min(pred)
        pred = (pred-mind)/(maxd-mind)
        # to numpy
        predict = pred.squeeze()
        predict_np = predict.cpu().data.numpy()
        im = cv2.cvtColor(predict_np*255, cv2.COLOR_GRAY2BGR)

        Qimage = cv2.imread(os.path.join(image_dir,img_name_list[i_test]))

        imo = cv2.resize(im, (Qimage.shape[1],Qimage.shape[0]),interpolation=cv2.INTER_LINEAR)
        tht,imo2 = cv2.threshold(imo, 9, 255, cv2.THRESH_BINARY)
        tht,imo3 = cv2.threshold(imo, 9, 255, cv2.THRESH_BINARY_INV)
        imo2 = imo2 / 255
        img_domain = np.multiply(Qimage, imo2)
        img_domain = img_domain + imo3
        cv2.imwrite(os.path.join(predi_dir,img_name_list[i_test]), img_domain)
        i+=1
    
    return img_domain

def preSOD(image_dir,predi_dir='./'):
    img_name_list=os.listdir(image_dir)
    img_name_list.sort()
    
    soddataset = SalObjDataset(img_name_list = img_name_list,image_dir=image_dir,transform=transforms.Compose([RescaleT(256),ToTensor()]))
    SODdataloader = DataLoader(soddataset, batch_size=1,shuffle=False,num_workers=0)
    
    SODdatasets(SODNet, SODdataloader,predi_dir,img_name_list,image_dir)

# if __name__ == "__main__":
#     preSOD(image_dir='./image',predi_dir='./preimage')
