import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from torchvision import transforms
from training_dataset import retrieval_dataset
import joblib

import net

import numpy as np

import os
from tqdm import tqdm



transform_480 = transforms.Compose([
    transforms.Resize((480,480)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

model_path={
    'densenet201':'./pretrained/densenet201.t7',
    'seresnet152':'./pretrained/seresnet152.t7',
}
feature_length={
    'densenet201':1920,
    'seresnet152':2048,
}


def get_feat(image_dir,model_name='seresnet152',batch_size=100,mode='cpu'): # or 'cuda'
    name_list=os.listdir(image_dir)
    name_list.sort()
    print(model_name)
    model=net.__dict__[model_name](model_path[model_name])
    if mode=='cuda':
        model=model.cuda()
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    dataset = retrieval_dataset(image_dir,transform=transform_480)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    if model_name =='seresnet152':
        feat_dict={
            'Grmac':torch.empty(len(name_list),feature_length[model_name]).half(),
        }
    else:
        feat_dict={
            'Mac':torch.empty(len(name_list),feature_length[model_name]).half()
        }
    img_list=[]
    model.eval()
    with torch.no_grad():
        for i, (inputs, names) in tqdm(enumerate(testloader)):
            if mode=='cuda':
                inputs = inputs.to(mode)
            if model_name =='seresnet152':
                feature_Grmac= model(inputs)
                feat_dict['Grmac'][i*batch_size:i*batch_size+len(names),:]=feature_Grmac.half().cpu()
            else:
                feature_Mac= model(inputs)
                feat_dict['Mac'][i*batch_size:i*batch_size+len(names),:]=feature_Mac.half().cpu()
            assert name_list[i*batch_size:i*batch_size+len(names)]==list(names)
            img_list.extend(names)
    testloader=None
    return img_list,feat_dict
