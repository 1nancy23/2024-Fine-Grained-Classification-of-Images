import os
from Model import EVCBlock
from mobilenetv3_seg import MobileNetV3Seg
import torch
import torchmetrics
from Data_in import Dataset1, Dataset2, Dataset3
from torch import nn
# from came import CAME
# model1=cspdarknet53(num_classes=26).cuda()
# model1=cspdarknet53(num_classes=200).cuda()
# weights_dict=torch.load('./weights/Model_niao99.pth')
# del_keys = []
# for k in del_keys:
#     del weights_dict[k]      
# model1.load_state_dict(weights_dict, strict=False)
print("load.Fi")
# from torchvision.models import resnet50
# from thop import profile

# flops, params = profile(model1, inputs=(1, 3, 224,224))
# print('FLOPs = ' + str(flops/1000**3) + 'G')
# print('Params = ' + str(params/1000**2) + 'M')
model1=MobileNetV3Seg(nclass=200).cuda()
model1.load_state_dict(torch.load(f"./weights/Model_CUB100_512_3.pth"))
# weights_dict=torch.load("./weights/Model_niao439.pth")
# del_keys = ["classifier1.2.weight","classifier1.2.bias",
#            "classifier2.2.weight","classifier2.2.bias",
#            "classifier3.2.weight","classifier3.2.bias",
#            "Att.fc.0.weight","Att.fc.2.weight",
#            "Att.W_Q.weight","Att.W_K.weight",
#            "Att.W_V.weight"]
# for k in del_keys:
#     del weights_dict[k]      
# model1.load_state_dict(weights_dict, strict=False)
# print(model1)
Loss1 = nn.CrossEntropyLoss()
optm1 = torch.optim.SGD(model1.parameters(), lr=0.001,weight_decay=0.001)
# scher = torch.optim.lr_scheduler.ReduceLROnPlateau(optm1, mode='min', patience=5, eps=1e-7)
scher=torch.optim.lr_scheduler.StepLR(optm1,20,0.5)
Dataset3 = Dataset3
Dataset1 = Dataset1
Dataset2 = Dataset2


def Train(Model, Loss, Optm):
    k=5
    score=0
    for x in range(100):
        # torch.cuda.empty_cache()
        model1.train()
        for i, (img, target) in enumerate(Dataset1):
            i += 1
            img = img.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            optm1.zero_grad()
            # y_pred = model1(img)
            # loss = Loss1(y_pred, target)
            # loss.backward()
            # optm.step()
            # NEW
            # with torch.cuda.amp.autocast():
            y_pred = model1(img)
            # print(y_pred.shape)
            # print(y_pred.shape,target.shape)
            loss = Loss1(y_pred, target)
            loss.backward()
            # NEW
            # scaler.scale(loss).backward()
            # loss.requires_grad_(True)
            lv = loss.item()
            print(f"Epoch {x + 1}/{100}; Batch {i}; Loss {lv}")
            # NEW
            # scaler.step(optm1)
            # scaler.update()
            optm1.step()
        scher.step()
        torch.cuda.empty_cache()
        if (x+1)%1==0:
            # scher.step(loss)
            model1.eval()
            test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=200).cuda()
            # torch.cuda.empty_cache()
            for i, (img, target) in enumerate(Dataset2):
                img = img.cuda(non_blocking=True)
                # print(img)
                target = target.cuda(non_blocking=True)
                pred = model1(img)
                test_acc(pred.argmax(1), target)
            torch.save(model1.state_dict(), f"./weights/Model_CUB100_512_4.pth")
            print("test", test_acc.compute())
            # with open(os.path.join('./result.txt'),'at') as f:
            #     f.write(f"./weights/Model_niao{400+k}.pth:\n")
            # with open(os.path.join('./result.txt'),'at') as f:
            #     f.write('test{}\n'.format(test_acc.compute()))

            # train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=200).cuda()
            # # torch.cuda.empty_cache()
            # for i, (img, target) in enumerate(Dataset3):
            #     img = img.cuda(non_blocking=True)
            #     # print(img)
            #     target = target.cuda(non_blocking=True)
            #     pred = model1(img)
            #     train_acc(pred.argmax(1), target)
            # print("test_val", train_acc.compute())
            #     f.write('test_val{}\n'.format(train_acc.compute()))
            # if train_acc.compute()>score:
            #     torch.save(model1.state_dict(), f"./weights/Model_niao500.pth")
            #     score=train_acc.compute()
            # k=k+1
Train(model1,Loss1,optm1)




