"""
@author: Yakhyokhuja Valikhujaev <yakhyo9696@gmail.com>
"""
from torch import  nn
import torch


class GlobalAvgPool2d(torch.nn.Module):
    def __init__(self,Size,Dim):
        super(GlobalAvgPool2d, self).__init__()
        self.TO1=nn.AvgPool2d(kernel_size=Size)
        self.DIM=Dim
    def forward(self,x):
        x=self.TO1(x)
        x=x.view(-1,self.DIM)
        return x



def auto_pad(k, p=None):
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p