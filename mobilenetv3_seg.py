"""MobileNet3 for Semantic Segmentation"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from base import BaseModel

__all__ = ['MobileNetV3Seg', 'get_mobilenet_v3_large_seg', 'get_mobilenet_v3_small_seg']


class MobileNetV3Seg(BaseModel):
    def __init__(self, nclass, aux=False, backbone='mobilenetv3_small', pretrained_base=False, **kwargs):
        super(MobileNetV3Seg, self).__init__(nclass, aux, backbone, pretrained_base, **kwargs)
        mode = backbone.split('_')[-1]
        # self.head = _Head(nclass, mode, **kwargs)
        if aux:
            inter_channels = 40 if mode == 'large' else 24
            self.auxlayer = nn.Conv2d(inter_channels, nclass, 1)
        self.Necks=Necks()
        self.head=ScaledDotProductAttention(nclass)
        self.classifier1 = nn.Sequential(
            GlobalAvgPool2d(8, 576),
            nn.ReLU(True),
            nn.Linear(576, nclass)
        )
        self.classifier2 = nn.Sequential(
            GlobalAvgPool2d(16, 96),
            nn.ReLU(True),
            nn.Linear(96, nclass),
        )
        self.classifier3 = nn.Sequential(
            GlobalAvgPool2d(32, 48),
            nn.ReLU(True),
            nn.Linear(48, nclass),
        )
    def forward(self, x):
        size = x.size()[2:]
        _1, c2, _2, c4 = self.base_forward(x)
        outputs = [c2,_2,c4]
        # x = self.head(c4)
        # x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs=self.Necks(outputs)
        outputs=self.head(self.classifier1(outputs[0]),self.classifier2(outputs[1]),self.classifier3(outputs[2]))
        if self.aux:
            auxout = self.auxlayer(c2)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return outputs


class _Head(nn.Module):
    def __init__(self, nclass, mode='small', norm_layer=nn.BatchNorm2d, **kwargs):
        super(_Head, self).__init__()
        in_channels = 960 if mode == 'large' else 576
        self.lr_aspp = _LRASPP(in_channels, norm_layer, **kwargs)
        self.project = nn.Conv2d(128, nclass, 1)

    def forward(self, x):
        x = self.lr_aspp(x)
        return self.project(x)
class Necks(nn.Module):
    def __init__(self):
        super(Necks,self).__init__()
        ###Ori=256,512,1024
        ###New=48,96,576
        self.EVC1_1 = nn.Sequential(
            # SpatialAttention(kernel_size=7),
            nn.Conv2d(48 + 96, 48, 5, 1, 2),
            EVCBlock(48, 48),
        )
        self.EVC1_2 = nn.Sequential(
            # SpatialAttention(kernel_size=7),
            nn.Conv2d(96 + 576, 96, 5, 1, 2),
            EVCBlock(96, 96),
        )
        self.EVC2_1 = nn.Sequential(
            # ChannelAttention(512+256,512),
            nn.Conv2d(48+96, 96, 3, 1, 1),
            EVCBlock(96, 96),
        )
        self.EVC2_2 = nn.Sequential(
            # ChannelAttention(512+1024),
            nn.Conv2d(96+576, 576, 3, 1, 1),
            EVCBlock(576, 576),
        )
        self.Conv1 = nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1)
        self.Conv2 = nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1)
        # self.UP2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.t_conv1 = nn.ConvTranspose2d(in_channels=576, out_channels=576, stride=2, kernel_size=5, padding=2,
                                          output_padding=1, dilation=1, padding_mode="zeros", bias=False)
        self.t_conv2 = nn.ConvTranspose2d(in_channels=96, out_channels=96, stride=2, kernel_size=5, padding=2,
                                          output_padding=1, dilation=1, padding_mode="zeros", bias=False)

        # self.Conv1=nn.Conv2d(64,64,kernel_size=3,stride=2,padding=1)
        # self.Conv2=nn.Conv2d(128,128,kernel_size=3,stride=2,padding=1)
        # self.UP2=nn.Upsample(scale_factor=2, mode='bilinear')
        # self.t_conv1 = nn.ConvTranspose2d(in_channels=256, out_channels=256, stride=2, kernel_size=5, padding=2, output_padding=1,dilation=1, padding_mode="zeros", bias=False)
        # self.t_conv2 = nn.ConvTranspose2d(in_channels=128, out_channels=128, stride=2, kernel_size=5, padding=2, output_padding=1,dilation=1, padding_mode="zeros", bias=False)
        # self.t_conv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        # self.t_conv4 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        # self.t_conv5 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)

        self.channelatten1 = ChannelAttention(576)
        self.spitialatten1 = SpatialAttention()
        self.channelatten2 = ChannelAttention(96)
        self.spitialatten2 = SpatialAttention()
        self.channelatten3 = ChannelAttention(48)
        self.spitialatten3 = SpatialAttention()
        self.channelatten4 = ChannelAttention(96)
        self.spitialatten4 = SpatialAttention()


    def forward(self,features):
        P1,P2,P3=features[0],features[1],features[2]## channels=64,128,256
        P3_to_2=self.t_conv1(P3)
        P3_to_2=self.channelatten1(P3_to_2)
        P3_to_2=self.spitialatten1(P3_to_2)
        P2_1=self.EVC1_2(torch.cat((P3_to_2,P2),dim=1))# c=512
        P2_to_1=self.t_conv2(P2_1)
        P2_to_1=self.channelatten2(P2_to_1)
        P2_to_1=self.spitialatten2(P2_to_1)
        P1_1=self.EVC1_1(torch.cat((P2_to_1,P1),dim=1))# c=256
        P1_to_2=self.Conv1(P1_1)
        P1_to_2=self.channelatten3(P1_to_2)
        P1_to_2=self.spitialatten3(P1_to_2)
        P2_2=self.EVC2_1(torch.cat((P1_to_2,P2_1),dim=1))# c=1024
        P2_to_3=self.Conv2(P2_2)
        P2_to_3=self.channelatten4(P2_to_3)
        P2_to_3=self.spitialatten4(P2_to_3)
        P3_1=self.EVC2_2(torch.cat((P2_to_3,P3),dim=1))

        return [P3_1,P2_2,P1_1]

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model):
        super(ScaledDotProductAttention, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, d_model, bias=False))
        self.d_model = d_model
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.Norm1 = nn.LayerNorm(self.d_model)
        self.Norm2 = nn.LayerNorm(self.d_model)

    def forward(self, Q, K, V):
        residual_1 = Q
        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
            self.d_model)  # scores : [batch_size, n_heads, len_q, len_k]
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        context = self.Norm1(context + residual_1)
        residual_2 = context
        output = self.fc(context)
        # print(output.shape)
        return self.Norm2(output + residual_2)  # [batch_size, seq_len, d_model]
class GlobalAvgPool2d(torch.nn.Module):
    def __init__(self,Size,Dim):
        super(GlobalAvgPool2d, self).__init__()
        self.TO1=nn.AvgPool2d(kernel_size=Size)
        self.DIM=Dim
    def forward(self,x):
        x=self.TO1(x)
        x=x.view(-1,self.DIM)
        return x


class _LRASPP(nn.Module):
    """Lite R-ASPP"""

    def __init__(self, in_channels, norm_layer, **kwargs):
        super(_LRASPP, self).__init__()
        out_channels = 128
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )
        self.b1 = nn.Sequential(
            # nn.AvgPool2d(kernel_size=(49, 49), stride=(16, 20)),  # check it
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat2 = F.interpolate(feat2, size, mode='bilinear', align_corners=True)
        x = feat1 * feat2  # check it
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x) * x1
from Model import EVCBlock

def get_mobilenet_v3_large_seg(num_classes, pretrained=False, root='~/.torch/models',
                               pretrained_base=False, **kwargs):
    acronyms = {
        'pascal_voc': 'pascal_voc',
        'pascal_aug': 'pascal_aug',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    # from light.data import datasets
    model = MobileNetV3Seg(nclass=num_classes, backbone='mobilenetv3_large',
                           pretrained_base=pretrained_base, **kwargs)
    # if pretrained:
    #     from ..model import get_model_file
    #     model.load_state_dict(
    #         torch.load(get_model_file('mobilenetv3_large_%s_best_model' % (acronyms[dataset]), root=root)))
    return model


def get_mobilenet_v3_small_seg(num_classes=2, pretrained=False, root='~/.torch/models',
                               pretrained_base=False, **kwargs):
    # acronyms = {
    #     'pascal_voc': 'pascal_voc',
    #     'pascal_aug': 'pascal_aug',
    #     'ade20k': 'ade',
    #     'coco': 'coco',
    #     'citys': 'citys',
    # }
    # from data import datasets
    model = MobileNetV3Seg(num_classes, backbone='mobilenetv3_small',
                           pretrained_base=pretrained_base, **kwargs)
    # if pretrained:
    #     from ..model import get_model_file
    #     model.load_state_dict(
    #         torch.load(get_model_file('mobilenetv3_small_%s_best_model' % (acronyms[dataset]), root=root)))
    return model


if __name__ == '__main__':
    model = get_mobilenet_v3_small_seg()
    Tensor=torch.randn(8,3,512,512)
    print(model(Tensor).shape)
    # for i in model(Tensor):
        # print(i.shape)