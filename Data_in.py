import cv2
import torch
from torch.utils.data import DataLoader,RandomSampler
from torchvision.transforms import transforms
from PIL import Image
from torch import Tensor as Ten
import numpy as np
import os
from torch import nn
import math
device='cuda'
Trans1=transforms.Compose(
            [
                # transforms.ColorJitter(brightness=.5,hue=.5),
                # transforms.RandomPerspective(p=0.5,distortion_scale=0.8),
                transforms.ToTensor(),
                transforms.RandomResizedCrop(512,(0.9,0.9)),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomApply(transforms)
                ]
        )
# Trans2=transforms.Compose(
#     [
#         # transforms.ToTensor(),
#         transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5),inplace=True)
#     ]
# )
class Dataset():
    def __init__(self,path_dir,Model):
        self.Blur1=transforms.GaussianBlur(kernel_size=7,sigma=(10,15))
        self.Blur2=transforms.GaussianBlur(kernel_size=7,sigma=(20,25))
        self.path_dir=path_dir
        self.list_class=os.listdir(path_dir)
        print(self.list_class)
        self.list_dir=[]
        self.target=[]
        Class=0
        if Model=="Train":
            for i in self.list_class:
                path_temp = os.listdir(path_dir + "/" + i)
                for x in path_temp[:int(len(path_temp)*0.5)]:
                    self.list_dir.append(path_dir + "/" + i+"/"+x)
                    self.target.append(Class)
                Class+=1
        if Model=="Val":
            for i in self.list_class:
                path_temp=os.listdir(path_dir + "/" + i)
                for x in path_temp[int(len(path_temp)*0.9):]:
                    self.list_dir.append(path_dir + "/" + i+"/"+x)
                    self.target.append(Class)
                Class+=1
        if Model=="T_Val":
            for i in self.list_class:
                path_temp=os.listdir(path_dir + "/" + i)[2:10]
                for x in path_temp:
                    self.list_dir.append(path_dir + "/" + i+"/"+x)
                    self.target.append(Class)
                Class+=1
        pass
    def __len__(self):
        # print("Data_Len",len(self.target))
        return len(self.target)
        pass
    def __getitem__(self, item):
        img=Image.open(self.list_dir[item])
        img=img.convert('RGB')
        img=(Trans1(img))
        target=self.target[item]
        return img,target
        pass
batch_size = 1
# class_sample_count = [444, 1014, 453, 569, 3451,626,792,560] # dataset has 10 class-1 samples, 1 class-2 samples, etc.
# weights = 2 / torch.Tensor(class_sample_count)
# sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, batch_size) # 注意这里的weights应为所有样本的权重序列，其长度为所有样本长度。

Dataset1=DataLoader(Dataset(r"D:\DATAS\CUB_200_2011\CUB_200_2011\images","Train"),batch_size=11,shuffle=True)
Dataset2=DataLoader(Dataset(r"D:\DATAS\CUB_200_2011\CUB_200_2011\images","Val"),shuffle=True,batch_size=5)
Dataset3=DataLoader(Dataset(r"D:\DATAS\CUB_200_2011\CUB_200_2011\images","T_Val"),shuffle=True,batch_size=5)
