import os
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from MainNet.models.condition.architecture import get_conv2d_layer
from MainNet.models.condition.Math_Module import P,Q
from MainNet.models.condition.restoration import HalfDnCNNSE
from MainNet.models.condition.utils import *

class Decom(nn.Module):
    def __init__(self):
        super().__init__()
        self.decom = nn.Sequential(
            get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            get_conv2d_layer(in_c=32, out_c=32, k=3, s=1, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            get_conv2d_layer(in_c=32, out_c=32, k=3, s=1, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            get_conv2d_layer(in_c=32, out_c=4, k=3, s=1, p=1),
            nn.ReLU()
        )
    def forward(self, input):
        output = self.decom(input)
        R = output[:, 0:3, :, :]
        L = output[:, 3:4, :, :]
        return R, L
class Inference_low(nn.Module):
    def __init__(self):
        super().__init__()
        # loading decomposition model
        self.model_Decom_low = Decom()
        checkpoint_Decom_low = torch.load('D:/Github-Code/BMNet-main/MainNet/pretrain/init_low.pth')
        self.model_Decom_low.load_state_dict(checkpoint_Decom_low['state_dict']['model_R'])
        # loading R; old_model_opts; and L model
    def forward(self, input_low_img):
        with torch.no_grad():
            RL,LL = self.model_Decom_low(input_low_img)
        return  RL,LL

class Inference_high(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_Decom_high = Decom()
        checkpoint_Decom_high = torch.load('D:/Github-Code/BMNet-main/MainNet/pretrain/init_high.pth')
        self.model_Decom_high.load_state_dict(checkpoint_Decom_high['state_dict']['model_R'])
    def forward(self, input_high_img):
        with torch.no_grad():
            RH, LH = self.model_Decom_high(input_high_img)
        return RH,LH
if __name__ == '__main__':
    x=torch.randn(1,3,256,256)
    y=torch.randn(1,3,256,256)
    model = Inference()
    # model.cuda()
    output = model(x)
    out = model(y)
    print(output.shape)
    print(out.shape)