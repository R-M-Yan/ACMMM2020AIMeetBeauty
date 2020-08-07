import torch
import torch.nn as nn

from pooling import *
from module.se_resnet import se_resnet152
from module.densenet import densenet201 as Densenet201

__all__=['L2N','densenet201','seresnet152']

#---------------feature extraction------------------#

class L2N(nn.Module):

    def __init__(self, eps=1e-6):
        super(L2N,self).__init__()
        self.eps = eps

    def forward(self, x):
        return x / (torch.norm(x, p=2, dim=1, keepdim=True) + self.eps).expand_as(x)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'eps=' + str(self.eps) + ')'

class seresnet152(nn.Module):
    def __init__(self,model_path):
        super(seresnet152, self).__init__()
        se152 = se_resnet152(pretrained=None)
        checkpoint=torch.load(model_path)
        se152.load_state_dict(checkpoint)
        self.norm=L2N()
        self.backbone=nn.Sequential(*list(se152.children())[:-2])
        self.Grmac=Grmac_Pooling(p=3.5)

    def forward(self,data):
        feature=self.backbone(data)
        feature_Grmac=self.norm(self.Grmac(feature))
        return feature_Grmac

class densenet201(nn.Module):
    def __init__(self,model_path):
        super(densenet201, self).__init__()
        dense201 =Densenet201()
        checkpoint=torch.load(model_path)
        dense201.load_state_dict(checkpoint)
        self.norm=L2N()
        self.backbone=nn.Sequential(*list(dense201.children())[:-1])
        self.Mac=Mac_Pooling()
        
    def forward(self,data):
        feature=self.backbone(data)
        feature_Mac=self.norm(self.Mac(feature))
        return feature_Mac