from math import exp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
import torchvision

from MainNet.models.archs.pvt_v2 import pvt_v2_b2_li
import torchvision.transforms as transforms
from MainNet.models.condition.condition_retinex import Inference_low,Inference_high
from MainNet.models.condition.condition_edge import edge_net
from MainNet.models.condition.condition_edge import FullGenerator
from MainNet.models.archs.color import ConditionNet
from MainNet.models.archs.edge import c_net
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class CoupleLayer(nn.Module):
    def __init__(self, channels, substructor, condition_length,  clamp=5.):
        super().__init__()

        channels = channels
        # self.ndims = len(dims_in[0])
        self.split_len1 = channels
        self.split_len2 = channels

        self.clamp = clamp
        self.max_s = exp(clamp)
        self.min_s = exp(-clamp)

        self.conditional = False
        # condition_length = sub_len
        # self.shadowpre = nn.Sequential(
        #     nn.Conv2d(4, channels // 2, 3, 1, 1),
        #     nn.LeakyReLU(0.2))
        # self.shadowpro = ShadowProcess(channels // 2)


        self.s1 = substructor(self.split_len1 , self.split_len2*2)
        self.s2 = substructor(self.split_len2 , self.split_len1*2)


        self.CG1R = nn.Conv2d(3, 16, 1, 1, 0)
        self.CG1L = nn.Conv2d(1, 16, 1, 1, 0)


        self.CG2R = nn.Conv2d(16, 3, 1, 1, 0)
        self.CG2L = nn.Conv2d(16, 1, 1, 1, 0)


    def e(self, s):
        return torch.exp(self.clamp * 0.636 * torch.atan(s / self.clamp))

    def log_e(self, s):
        return self.clamp * 0.636 * torch.atan(s / self.clamp)

    def forward(self, R,L,edge,num, rev=False):
        R=self.CG1R(R)
        L=self.CG1L(L)

        if not rev:
            # r2 = self.s2(torch.cat([x2, c_star], 1) if self.conditional else x2)
            if num==1:
                r2 = self.s2(L,None)
            else:
                r2 = self.s2(L, None)

            # 产生s2=fai  t2=rou
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = self.e(s2) * R + t2


            r1 = self.s1(y1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]

            y2 = self.e(s1) * L + t1


        else: # names of x and y are swapped!

            r1 = self.s1(R)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = (L - t1) / self.e(s1)

            if num==1:
                r2 = self.s2(y2, None)
            else:
                r2 = self.s2(y2,None)

            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]

            y1 = (R - t2) / self.e(s2)

        y1 = self.CG2R(y1)
        y2 = self.CG2L(y2)

        return y1,y2

    # def jacobian(self, x, c=[], rev=False):
    #     return torch.sum(self.last_jac, dim=tuple(range(1, self.ndims+1)))
    def output_dims(self, input_dims):
        return input_dims


def upsample(x, h, w):
    return F.interpolate(x, size=[h, w], mode='bicubic', align_corners=True)

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.1, use_HIN=True):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        out = self.conv_1(x)

        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out


class ShadowProcess(nn.Module):
    def __init__(self, channels):
        super(ShadowProcess, self).__init__()
        self.process = UNetConvBlock(channels, channels)
        self.Attention = nn.Sequential(
            nn.Conv2d(channels,channels,3,1,1),
            nn.Sigmoid())

    def forward(self, x):
        x = self.process(x)
        xatt = self.Attention(x)

        return xatt

class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=16, bias=False):
        super(DenseBlock, self).__init__()
        self.conv1 = UNetConvBlock(channel_in, gc)
        self.conv2 = UNetConvBlock(gc, gc)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, channel_out, 1, 1, 0, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3], 0.1)
        # initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))

        return x3

class MultiscaleDense(nn.Module):
    def __init__(self,channel_in, channel_out, init):
        super(MultiscaleDense, self).__init__()
        self.conv_mul = nn.Conv2d(4,channel_out//2,3,1,1)
        self.conv_add = nn.Conv2d(4, channel_out//2, 3, 1, 1)

        self.down1 = nn.Conv2d(channel_out//2,channel_out//2,stride=2,kernel_size=2,padding=0)
        self.down2 = nn.Conv2d(channel_out//2, channel_out//2, stride=2, kernel_size=2, padding=0)
        self.op1 = DenseBlock(channel_in, channel_out//2, init)
        # self.op2 = DenseBlock(channel_in, channel_out, init)
        # self.op3 = DenseBlock(channel_in, channel_out, init)
        self.fuse = nn.Conv2d(2 * channel_out//2, channel_out, 1, 1, 0)

        self.pvt=pvt_v2_b2_li()
    def forward(self, x,cond=None):
        if cond is not None:
            alpha = self.conv_mul(cond)
            beta= self.conv_add(cond)
            x_op = alpha*x+beta
        else:
            x_op = x
        # x = torch.cat([x,x_trans],1)
        x1=self.op1(torch.cat([x_op],1))

        x_down= self.down1(x)

        if cond is not None:
            alpha_down,beta_down=F.interpolate(alpha, scale_factor=0.5, mode='bilinear'), F.interpolate(beta, scale_factor=0.5, mode='bilinear')
            xop_down=x_down*alpha_down+beta_down
        else:
            xop_down = x_down

        x_temp = self.pvt(xop_down, None)
        # x3,x4=x_temp[0],x_temp[1]
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x1 = self.op1(torch.cat([x1],1))
        # x2 = self.op2(torch.cat([x2],1))
        # x3 = self.op3(torch.cat([x3],1))
        # x3 = x_temp[1]
        # x3 = F.interpolate(x3, size=(x1.size()[2], x1.size()[3]), mode='bilinear')
        x_temp = F.interpolate(x_temp, size=(x1.size()[2], x1.size()[3]), mode='bilinear')
        x = self.fuse(torch.cat([x1, x_temp], 1))

        return x


def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out):
        if net_structure == 'DBNet':
            if init == 'xavier':
                return MultiscaleDense(channel_in, channel_out, init)
            else:
                return MultiscaleDense(channel_in, channel_out, init)
            # return UNetBlock(channel_in, channel_out)
        else:
            return None

    return constructor

class InvISPNet(nn.Module):
    def __init__(self, channel_in=3, subnet_constructor=subnet('DBNet'), block_num=4):
        super(InvISPNet, self).__init__()
        operations = []
        # level = 3
        # self.condition = ConditionNet()
        # self.condition.load_state_dict(torch.load('/home/jieh/Projects/Shadow/MainNet/pretrain/condition.pth'))
        for p in self.parameters():
            p.requires_grad = False

        channel_num = 16  # total channels at input stage

        self.color = ConditionNet()
        self.color.load_state_dict(torch.load('D:/Github_Code/BMNet-main/BMNet-main/MainNet/pretrain/1460_G.pth'))

        self.edge=c_net()
        ckpt = torch.load('D:/Github-Code/SMG-LLIE-main/SMG-LLIE-main/result/checkpointsnogan/iteration_437000.pt')
        self.edge.load_state_dict(ckpt['state_dict'])

        self.CG0R = nn.Conv2d(channel_in, channel_num, 1, 1, 0)
        self.CG0L = nn.Conv2d(channel_in // 3, channel_num, 1, 1, 0)


        self.CG1R = nn.Conv2d(channel_num, channel_in, 1, 1, 0)
        self.CG1L = nn.Conv2d(channel_num, channel_in // 3, 1, 1, 0)

        self.CG2R = nn.Conv2d(channel_in, channel_num, 1, 1, 0)
        self.CG2L = nn.Conv2d(channel_in // 3, channel_num, 1, 1, 0)

        self.CG3R = nn.Conv2d(channel_num, channel_in, 1, 1, 0)
        self.CG3L = nn.Conv2d(channel_num, channel_in // 3, 1, 1, 0)
        #
        self.conv_mul = nn.Conv2d(3,3, 1, 1, 0)
        self.conv_add = nn.Conv2d(3,3,1,1,0)

        for j in range(3):
            b = CoupleLayer(channel_num, substructor = subnet_constructor, condition_length=channel_num//2)  # one block is one flow step.
            operations.append(b)

        self.operations = nn.ModuleList(operations)
        self.initialize()
        self.decomL = Inference_low()

        self.clamp=5.
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= 1.  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= 1.
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)
    def e(self, s):
        return torch.exp(self.clamp * 0.636 * torch.atan(s / self.clamp))
    def forward(self, input,gt, rev=False):
        color_cond=self.color(input)

        x_gray = input[:, 0:1, :, :] * 0.299 + input[:, 1:2, :, :] * 0.587 + input[:, 2:3, :, :] * 0.114
        edge_cond = self.edge(x_gray)

        cond = torch.cat([color_cond,edge_cond],dim=1)

        if not rev:
            R,L=self.decomL(input)

            num = 0
            for op in self.operations:
                num=num+1
                R,L = op.forward(R,L,cond,num, rev)

            return R,L,cond
        else:
            R, L = self.decomL(gt)

            num=5
            for op in reversed(self.operations):
                num=num-1
                R,L = op.forward(R,L,cond,num, rev)

            return R,L



if __name__ == '__main__':
    level =3
    net = InvISPNet(channel_in=3,block_num=8)
    print('#generator parameters:',sum(param.numel() for param in net.parameters()))
    x = torch.randn(1, 3, 256, 256)

    print(x.size())
    out = net(x,gt=1,rev=False)
    print(out.shape)
