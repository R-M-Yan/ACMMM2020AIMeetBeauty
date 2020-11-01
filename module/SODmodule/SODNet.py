#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SODNet model.
@author: Yanrunming
"""

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from .unitFunc_model import *
from efficientnet_pytorch import EfficientNet


class SODNet(nn.Module):
    def __init__(self,n_channels,n_classes):
        super(SODNet,self).__init__()

        effinet = EfficientNet.from_pretrained('efficientnet-b0')
        # MBConvX(inp, oup, stride, expand_ratio, kernel)
        
        # stage1        
        self.s1conv1 = Conv_3x3(n_channels, 32, 1)  # 224*224
        self.s1conv2 = Conv_3x3(32, 32, 1)  # 224*224
        self.s1conv3 = Conv_3x3(32, 32, 1)  # 224*224
        
        self.s1dconv = effinet._conv_stem  # out: 112*112 32 
        self.s1dconvbn = effinet._bn0

        # stage2
        self.s2conv1 = effinet._blocks[0]  # out: 112*112 16          
        self.s2dconv = effinet._blocks[1]  # out: 56*56 24 
 
        # stage3       
        self.s3conv1 = effinet._blocks[2]  # out: 56*56 24 
        self.s3dconv = effinet._blocks[3]  # out: 28*28 40        
        
        # stage4
        self.s4conv1 = effinet._blocks[4]  # out: 28*28 40     
        self.s4dconv = effinet._blocks[5]  # out: 14*14 80          
                
        # stage5
        self.s5conv1 = effinet._blocks[6]  # out: 14*14 80   
        self.s5conv2 = effinet._blocks[7]  # out: 14*14 80    
        
        self.s5conv3 = effinet._blocks[8]  # out: 14*14 112 
        self.s5conv4 = effinet._blocks[9]  # out: 14*14 112  
        self.s5conv5 = effinet._blocks[10]  # out: 14*14 112          
        self.s5dconv = effinet._blocks[11]  # out: 7*7 192  
        
        # stage6
        self.s6conv1 = effinet._blocks[12]  # out: 7*7 192   
        self.s6conv2 = effinet._blocks[13]  # out: 7*7 192         
        self.s6conv3 = effinet._blocks[14]  # out: 7*7 192 
        
        self.s6conv4 = effinet._blocks[15]  # out: 7*7 320                     
        
        # decoder stage6
        self.dec_block6 = DecoBlock(320,320)  
        self.dec_block6up = DecoBlockWithUpsamp(320,320)   

        self.debasic6_1 = BasicBlock(320,320)           
        self.debasic6_2 = BasicBlock(320,320)  
        
        # decoder stage5
        self.dec_block5 = DecoBlock(432,112)  
        self.dec_block5up = DecoBlockWithUpsamp(112,112)           

        self.debasic5_1 = BasicBlock(112,112)  
        self.debasic5_2 = BasicBlock(112,112)  
        
        # decoder stage4
        self.dec_block4 = DecoBlock(152,40)  
        self.dec_block4up = DecoBlockWithUpsamp(40,40)  

        self.debasic4_1 = BasicBlock(40,40)  
        self.debasic4_2 = BasicBlock(40,40)  
        
        # decoder stage3
        self.dec_block3 = DecoBlock(64,24)  
        self.dec_block3up = DecoBlockWithUpsamp(24,24)          

        self.debasic3_1 = BasicBlock(24,24)  
        self.debasic3_2 = BasicBlock(24,24)  
        
        # decoder stage2
        self.dec_block2 = DecoBlock(40,16)  
        self.dec_block2up = DecoBlockWithUpsamp(16,16) 

        self.debasic2_1 = BasicBlock(16,16)    
          
        # decoder stage1
        self.dec_block1 = DecoBlock(48,32)  
        self.supconv1 = nn.Conv2d(32,1,3,padding=1)
        
        self.debasic1_1 = BasicBlock(32,32)  


        self.outconv = nn.Conv2d(544, 1, 1, 1, 0, bias=False)

        ## -------------Bilinear Upsampling--------------
        self.supscore6 = nn.Upsample(scale_factor=32,mode='bilinear')
        self.supscore5 = nn.Upsample(scale_factor=16,mode='bilinear')
        self.supscore4 = nn.Upsample(scale_factor=8,mode='bilinear')
        self.supscore3 = nn.Upsample(scale_factor=4,mode='bilinear')
        self.supscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

        ## -------------Side Output--------------
        self.supconv6 = nn.Conv2d(320,1,3,padding=1)
        self.supconv5 = nn.Conv2d(112,1,3,padding=1)
        self.supconv4 = nn.Conv2d(40,1,3,padding=1)
        self.supconv3 = nn.Conv2d(24,1,3,padding=1)
        self.supconv2 = nn.Conv2d(16,1,3,padding=1)
          
    def forward(self,x):

        # stage1       
        s1_x1 = self.s1conv1(x) #224*224
        s1_x2 = self.s1conv2(s1_x1) #224*224 
        s1_x3 = self.s1conv3(s1_x2) #224*224  
        
        s1_xd = self.s1dconv(x)  # out: 112*112 32 
        s1_xdbn = self.s1dconvbn(s1_xd)           
        #print(s1_xdbn.shape)
        
        # stage2 
        s2_x1 = self.s2conv1(s1_xdbn)  # out: 112*112 16              
        s2_xd = self.s2dconv(s2_x1)  # out: 56*56 24          
        #print(s2_xd.shape)
        
        # stage3 
        s3_x1 = self.s3conv1(s2_xd)  # out: 56*56 24     
        s3_xd = self.s3dconv(s3_x1)  # out: 28*28 40   
       # print(s3_xd.shape)     
        
        # stage4 
        s4_x1 = self.s4conv1(s3_xd)  # out: 28*28 40           
        s4_xd = self.s4dconv(s4_x1)  # out: 14*14 80         
       # print(s4_xd.shape)
        
        # stage5 
        s5_x1 = self.s5conv1(s4_xd)  # out: 14*14 80     
        s5_x2 = self.s5conv2(s5_x1)  # out: 14*14 80   
        
        s5_x3 = self.s5conv3(s5_x2)  # out: 14*14 112  
        s5_x4 = self.s5conv4(s5_x3)  # out: 14*14 112  
        s5_x5 = self.s5conv5(s5_x4)  # out: 14*14 112          
        s5_xd = self.s5dconv(s5_x5)  # out: 7*7 192 
        #print(s5_xd.shape)
        
        # stage6
        s6_x1 = self.s6conv1(s5_xd)  # out: 7*7 192    
        s6_x2 = self.s6conv2(s6_x1)  # out: 7*7 192   
        s6_x3 = self.s6conv3(s6_x2)  # out: 7*7 192   
        s6_x4 = self.s6conv4(s6_x3)  # out: 7*7 320         
                      
        # decoder stage6
        s6_db = self.dec_block6(s6_x4)  
        s6_dbup, s6sup = self.dec_block6up(s6_db)        

        # decoder stage5
        s5_db = self.dec_block5(torch.cat((s5_x5,s6_dbup),1))  
        s5_dbup, s5sup = self.dec_block5up(s5_db)
       
        # decoder stage4
        s4_db = self.dec_block4(torch.cat((s4_x1,s5_dbup),1))  
        s4_dbup, s4sup= self.dec_block4up(s4_db)
        
        # decoder stage3
        s3_db = self.dec_block3(torch.cat((s3_x1,s4_dbup),1))  
        s3_dbup, s3sup = self.dec_block3up(s3_db)        

        # decoder stage2
        s2_db = self.dec_block2(torch.cat((s2_x1,s3_dbup),1))  
        s2_dbup, s2sup = self.dec_block2up(s2_db) 

        # decoder stage1
        s1_db = self.dec_block1(torch.cat((s1_x3,s2_dbup),1))  
        
        sup1 = self.supconv1(s1_db)
        
        # sup
        d6 = self.supconv6(s6sup)
        sup6 = self.supscore6(d6) # 7->224

        d5 = self.supconv5(s5sup)
        sup5 = self.supscore5(d5) # 14->256

        d4 = self.supconv4(s4sup)
        sup4 = self.supscore4(d4) # 28->256

        d3 = self.supconv3(s3sup)
        sup3 = self.supscore3(d3) # 56->256

        d2 = self.supconv2(s2sup)
        sup2 = self.supscore2(d2) # 112->256  
        
        # refine
        s6_ba1 = self.debasic6_1(s6sup)
        s6_ba2 = self.debasic6_2(s6_ba1)
        s66 = self.supscore6(s6_ba2) # 7->224
        s5_ba1 = self.debasic5_1(s5sup)
        s5_ba2 = self.debasic5_2(s5_ba1)
        s55 = self.supscore5(s5_ba2) # 14->256        
        s4_ba1 = self.debasic4_1(s4sup)
        s4_ba2 = self.debasic4_2(s4_ba1)  
        s44 = self.supscore4(s4_ba2) # 28->256        
        s3_ba1 = self.debasic3_1(s3sup)
        s3_ba2 = self.debasic3_2(s3_ba1)  
        s33 = self.supscore3(s3_ba2) # 56->256
        s2_ba1 = self.debasic2_1(s2sup)
        s22 = self.supscore2(s2_ba1) # 112->256        
        s1_ba1 = self.debasic1_1(s1_db)  
        out = self.outconv(torch.cat((s66,s55,s44,s33,s22,s1_ba1),1))
        
        return  F.sigmoid(out)
    
    