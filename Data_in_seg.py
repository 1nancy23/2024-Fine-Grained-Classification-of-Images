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
                transforms.Resize((512,512)),
                # transforms.RandomApply(transforms)
                ]
        )
Trans2=transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((256,256)),
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
        self.path_dir=path_dir
        self.list_dir=[]
        self.target=[]
        Class=0
        if Model=="Train":
            for i in os.listdir(path_dir+'/'+'image/'):
                self.list_dir.append(path_dir + "/image/"+i)
                i_2=os.path.splitext(i)[0].split('_')
                i_2.append('mask.png')
                if i_2[0]=='metal':
                    i_2[2]='ground_truth'
                else:
                    i_2[1]='ground_truth'
                self.target.append(path_dir+"/ground_truth/"+'_'.join(i_2))
        if Model=="Val":
            for i in os.listdir(path_dir + '/' + 'image/'):
                self.list_dir.append(path_dir + "/image/" + i)
                i_2=os.path.splitext(i)[0].split('_')
                i_2.append('mask.png')
                if i_2[0]=='metal':
                    i_2[2]='ground_truth'
                else:
                    i_2[1]='ground_truth'
                self.target.append(path_dir+"/ground_truth/"+'_'.join(i_2))
        if Model=="T_Val":
            for i in os.listdir(path_dir + '/' + 'image/'):
                self.list_dir.append(path_dir + "/image/" + i)
                i_2=os.path.splitext(i)[0].split('_')
                i_2.append('mask.png')
                if i_2[0]=='metal':
                    i_2[2]='ground_truth'
                else:
                    i_2[1]='ground_truth'
                self.target.append(path_dir+"/ground_truth/"+'_'.join(i_2))
        pass
    def __len__(self):
        # print("Data_Len",len(self.target))
        return len(self.target)
        pass
    def __getitem__(self, item):
        img=Image.open(self.list_dir[item])
        img=img.convert('RGB')
        img=(Trans1(img))
        target=Image.open(self.target[item])
        target=target.convert("1")
        target=Trans2(target)
        return img,target
        pass
# batch_size = 1
# class_sample_count = [444, 1014, 453, 569, 3451,626,792,560] # dataset has 10 class-1 samples, 1 class-2 samples, etc.
# weights = 2 / torch.Tensor(class_sample_count)
# sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, batch_size) # 注意这里的weights应为所有样本的权重序列，其长度为所有样本长度。

Dataset1=DataLoader(Dataset("G:/DATAS/my_mvtec_anomaly_detection","Train"),batch_size=1,shuffle=True)
Dataset2=DataLoader(Dataset("G:/DATAS/my_mvtec_anomaly_detection","Val"),shuffle=True,batch_size=1)
Dataset3=DataLoader(Dataset("G:/DATAS/my_mvtec_anomaly_detection","T_Val"),shuffle=True,batch_size=1)

# for i,(img,target) in enumerate(Dataset1):
#     print(img.shape,target.shape)
#     print(torch.max(img),torch.min(img),torch.max(target),torch.min(target))