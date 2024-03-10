"""
@author: Yakhyokhuja Valikhujaev <yakhyo9696@gmail.com>
"""
import torch
import torch.nn as nn
from torch.nn import MaxPool2d, functional as F
from utils import GlobalAvgPool2d, auto_pad
import numpy as np
import torch.utils.data as Data
__all__ = ['darknet19', 'darknet53', 'darknet53e', 'cspdarknet53']



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x.size() 30,40,50,30
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 30,1,50,30
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  # 30,1,50,30
        return self.sigmoid(x)  # 30,1,50,30


class UseAttentionModel(nn.Module):
    def __init__(self):
        super(UseAttentionModel, self).__init__()
        self.channel_attention = SpatialAttention()

    def forward(self, x):  # 反向传播
        attention_value = self.channel_attention(x)
        out = x.mul(attention_value)
        return out

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, auto_pad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.01) if act else nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, c1, shortcut=True):
        super(ResidualBlock, self).__init__()
        c2 = c1 // 2
        self.shortcut = shortcut
        self.layer1 = Conv(c1, c2, p=0)
        self.layer2 = Conv(c2, c1, k=3)

    def forward(self, x):
        residual = x
        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        if self.shortcut:
            out += residual
        return out


class CSP(nn.Module):
    """ [https://arxiv.org/pdf/1911.11929.pdf] """
    def __init__(self, c1, c2, num_blocks=1, shortcut=True, g=1, e=0.5):
        super(CSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.conv1 = Conv(c1, c_, 1, 1)
        self.conv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.conv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[ResidualBlock(c_, shortcut=shortcut) for _ in range(num_blocks)])

    def forward(self, x):
        y1 = self.conv3(self.m(self.conv1(x)))
        y2 = self.conv2(x)
        return self.conv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class Elastic(nn.Module):
    """ [https://arxiv.org/abs/1812.05262] """
    def __init__(self, c1):
        super(Elastic, self).__init__()
        c2 = c1 // 2

        self.down = nn.AvgPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.layer1 = Conv(c1, c2 // 2, p=0)
        self.layer2 = Conv(c2 // 2, c1, k=3)

    def forward(self, x):
        residual = x
        elastic = x

        # check the input size before downsample
        if x.size(2) % 2 > 0 or x.size(3) % 2 > 0:
            elastic = F.pad(elastic, (0, x.size(3) % 2, 0, x.size(2) % 2), mode='replicate')

        down = self.down(elastic)
        elastic = self.layer1(down)
        elastic = self.layer2(elastic)
        up = self.up(elastic)
        # check the output size after upsample
        if up.size(2) > x.size(2) or up.size(3) > x.size(3):
            up = up[:, :, :x.size(2), :x.size(3)]

        half = self.layer1(x)
        half = self.layer2(half)

        out = up + half  # elastic add
        out += residual  # residual add

        return out


class DarkNet19(nn.Module):
    """ [https://arxiv.org/pdf/1612.08242.pdf] """
    def __init__(self, num_classes=1000, init_weight=True):
        super(DarkNet19, self).__init__()

        if init_weight:
            self._initialize_weights()

        self.features = nn.Sequential(
            Conv(3, 32, 3),
            MaxPool2d(2, 2),

            Conv(32, 64, 3),
            MaxPool2d(2, 2),

            Conv(64, 128, 3),
            Conv(128, 64, 1),
            Conv(64, 128, 3),
            MaxPool2d(2, 2),

            Conv(128, 256, 3),
            Conv(256, 128, 1),
            Conv(128, 256, 3),
            MaxPool2d(2, 2),

            Conv(256, 512, 3),
            Conv(512, 256, 1),
            Conv(256, 512, 3),
            Conv(512, 256, 1),
            Conv(256, 512, 3),
            MaxPool2d(2, 2),

            Conv(512, 1024, 3),
            Conv(1024, 512, 1),
            Conv(512, 1024, 3),
            Conv(1024, 512, 1),
            Conv(512, 1024, 3),
        )

        self.classifier = nn.Sequential(
            *self.features,
            Conv(1024, num_classes, 1),
            GlobalAvgPool2d()
        )

    def forward(self, x):
        return self.classifier(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class DarkNet53(nn.Module):
    """ [https://pjreddie.com/media/files/papers/YOLOv3.pdf] """
    def __init__(self, block, num_classes=1000, init_weight=True):
        super(DarkNet53, self).__init__()
        self.num_classes = num_classes

        if init_weight:
            self._initialize_weights()

        self.features = nn.Sequential(
            Conv(3, 32, 3),

            Conv(32, 64, 3, 2),
            *self._make_layer(block, 64, num_blocks=1),

            Conv(64, 128, 3, 2),
            *self._make_layer(block, 128, num_blocks=2),

            Conv(128, 256, 3, 2),
            *self._make_layer(block, 256, num_blocks=8),

            Conv(256, 512, 3, 2),
            *self._make_layer(block, 512, num_blocks=8),

            Conv(512, 1024, 3, 2),
            *self._make_layer(block, 1024, num_blocks=4)
        )
        self.classifier = nn.Sequential(
            *self.features,
            GlobalAvgPool2d(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _make_layer(block, in_channels, num_blocks):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
 
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
 
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)*x
 
 
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
 
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
 
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x1 =x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)*x1

    
from Model import EVCBlock
class Necks(nn.Module):
    def __init__(self):
        super(Necks,self).__init__()
        self.EVC1_1=nn.Sequential(
            nn.Conv2d(64+128,64,5,1,2),
            EVCBlock(64,64),
        )
        self.EVC1_2=nn.Sequential(
            nn.Conv2d(256+128,128,5,1,2),
            EVCBlock(128,128),
        )
        self.EVC2_1=nn.Sequential(
            nn.Conv2d(128+64,128,3,1,1),
            EVCBlock(128,128),
        )    
        self.EVC2_2=nn.Sequential(
            nn.Conv2d(128+256,256,3,1,1),
            EVCBlock(256,256),
        )
        self.Conv1=nn.Conv2d(64,64,kernel_size=3,stride=2,padding=1)
        self.Conv2=nn.Conv2d(128,128,kernel_size=3,stride=2,padding=1)
        self.UP2=nn.Upsample(scale_factor=2, mode='bilinear')
        self.t_conv1 = nn.ConvTranspose2d(in_channels=256, out_channels=256, stride=2, kernel_size=5, padding=2, output_padding=1,dilation=1, padding_mode="zeros", bias=False)
        self.t_conv2 = nn.ConvTranspose2d(in_channels=128, out_channels=128, stride=2, kernel_size=5, padding=2, output_padding=1,dilation=1, padding_mode="zeros", bias=False)
        self.channelatten1=ChannelAttention(256)
        self.spitialatten1=SpatialAttention()
        self.channelatten2=ChannelAttention(128)
        self.spitialatten2=SpatialAttention()
        self.channelatten3=ChannelAttention(64)
        self.spitialatten3=SpatialAttention()
        self.channelatten4=ChannelAttention(128)
        self.spitialatten4=SpatialAttention()

    def forward(self,features):
        P1,P2,P3=features[0],features[1],features[2]## channels=64,128,256
        P3_to_2=self.t_conv1(P3)
        P3_to_2=self.channelatten1(P3_to_2)
        P3_to_2=self.spitialatten1(P3_to_2)
        P2_1=self.EVC1_2(torch.cat((P3_to_2,P2),dim=1))# c=128
        P2_to_1=self.t_conv2(P2_1)
        P2_to_1=self.channelatten2(P2_to_1)
        P2_to_1=self.spitialatten2(P2_to_1)
        P1_1=self.EVC1_1(torch.cat((P2_to_1,P1),dim=1))# c=64
        P1_to_2=self.Conv1(P1_1)
        P1_to_2=self.channelatten3(P1_to_2)
        P1_to_2=self.spitialatten3(P1_to_2)
        P2_2=self.EVC2_1(torch.cat((P1_to_2,P2_1),dim=1))# c=128
        P2_to_3=self.Conv2(P2_2)
        P2_to_3=self.channelatten4(P2_to_3)
        P2_to_3=self.spitialatten4(P2_to_3)
        P3_1=self.EVC2_2(torch.cat((P2_to_3,P3),dim=1))
        print('123',P1_1.shape,P3_1.shape,P2_2.shape)
        return [P2_2,P3_1,P1_1]
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self,d_model):
        super(ScaledDotProductAttention, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, d_model, bias=False))
        self.d_model=d_model
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.Norm1=nn.LayerNorm(self.d_model)
        self.Norm2=nn.LayerNorm(self.d_model)
    def forward(self, Q, K, V):
        residual_1=Q
        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_model)   # scores : [batch_size, n_heads, len_q, len_k]
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)                                # [batch_size, n_heads, len_q, d_v]
        context = nn.LayerNorm(self.d_model)(context + residual_1)
        residual_2 = context
        output = self.fc(context)
        return nn.LayerNorm(self.d_model)(output + residual_2)   # [batch_size, seq_len, d_model]
    
class CSPDarkNet53(nn.Module):
    """ [https://pjreddie.com/media/files/papers/YOLOv3.pdf] """
    def __init__(self, block, num_classes=1000, init_weight=True):
        super(CSPDarkNet53, self).__init__()
        self.num_classes = num_classes

        if init_weight:
            self._initialize_weights()

        self.features1 = nn.Sequential(
            Conv(3, 32, 3),

            Conv(32, 64, 3, 2),
            block(64, 64, num_blocks=1),
            Conv(64, 128, 3, 1),
            block(128, 128, num_blocks=2),
            Conv(128, 256, 3, 1),
            block(256, 256, num_blocks=4),
            block(256,64,num_blocks=1),
        )
        256#
        self.features2=nn.Sequential(
            Conv(64, 128, 3, 2),
            block(128, 128, num_blocks=4),
        )
        512#
        self.features3=nn.Sequential(
            Conv(128, 256, 3, 2),
            block(256, 256, num_blocks=4),
        )
        1024#
        self.classifier1 = nn.Sequential(
            GlobalAvgPool2d(64,256),
            nn.ReLU(True),
            nn.Linear(256, num_classes)
        )
        self.classifier2=nn.Sequential(
            GlobalAvgPool2d(128,128),
            nn.ReLU(True),
            nn.Linear(128,num_classes),
        )
        self.classifier3=nn.Sequential(
            GlobalAvgPool2d(256,64),
            nn.ReLU(True),
            nn.Linear(64,num_classes),
        )
        self.Neckm=Necks()
        self.Att=ScaledDotProductAttention(num_classes)
        self.Change=nn.Linear(num_classes,64)
    def forward(self, x):
        x1=self.features1(x)
        x2=self.features2(x1)
        x3=self.features3(x2)
        print(x1.shape,x2.shape,x3.shape)
        Features=self.Neckm([x1,x2,x3])###channels=64,128,256
        return self.Att(self.classifier1(Features[1]),self.classifier2(Features[0]),self.classifier3(Features[2]))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def darknet19(num_classes=1000, init_weight=True):
    return DarkNet19(num_classes=num_classes, init_weight=init_weight)


def darknet53(num_classes=1000, init_weight=True):
    return DarkNet53(ResidualBlock, num_classes=num_classes, init_weight=init_weight)


def darknet53e(num_classes=1000, init_weight=True):
    """ DarkNet53 with ELASTIC block """
    return DarkNet53(Elastic, num_classes=num_classes, init_weight=init_weight)


def cspdarknet53(num_classes=1000, init_weight=True):
    """ DarkNet53 with CSP block """
    return CSPDarkNet53(CSP, num_classes=num_classes, init_weight=init_weight)


# if __name__ == '__main__':
#     x = torch.randn(1, 3, 224, 224)
#
#     darknet19 = darknet19()
#     darknet19_features = darknet19.features
#
#     darknet53 = darknet53()
#     darknet53_features = darknet53.features
#
#     darknet53e = darknet53e()
#     darknet53e_features = darknet53e.features
#
#     cspdarknet53 = cspdarknet53()
#     cspdarknet53_features = cspdarknet53.features
#
#     print('Num. of Params of DarkNet19: {}'.format(sum(p.numel() for p in darknet19.parameters() if p.requires_grad)))
#     print('Num. of Params of DarkNet53: {}'.format(sum(p.numel() for p in darknet53.parameters() if p.requires_grad)))
#     print('Num. of Params of DarkNet53-ELASTIC: {}'.format(sum(p.numel() for p in darknet53e.parameters() if p.requires_grad)))
#     print('Num. of Params of CSP-DarkNet53: {}'.format(sum(p.numel() for p in cspdarknet53.parameters() if p.requires_grad)))
#
#     print('Output of DarkNet19: {}'.format(darknet19(x).shape))
#     print('Output of DarkNet53: {}'.format(darknet53(x).shape))
#     print('Output of Elastic DarkNet53-ELASTIC: {}'.format(darknet53e(x).shape))
#     print('Output of CSP-DarkNet53: {}'.format(cspdarknet53(x).shape))
#
#     print('Feature Extractor Output of DarkNet19: {}'.format(darknet19_features(x).shape))
#     print('Feature Extractor Output of DarkNet53: {}'.format(darknet53_features(x).shape))
#     print('Feature Extractor Output of DarkNet53-ELASTIC: {}'.format(darknet53e_features(x).shape))
#     print('Feature Extractor Output of CSP-DarkNet53: {}'.format(cspdarknet53_features(x).shape))
Model1=cspdarknet53(num_classes=256)
Tensor1=torch.randn(1,3,512,512)
Tensor3=Model1(Tensor1)
torch.save(Model1.state_dict(),"12412.pth")
print(Tensor3.shape)

# 定义一个类，名为 ActivationsAndGradients
class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """
    # 定义类的初始化方法，接受三个参数：model, target_layers, reshape_transform
    def __init__(self, model, target_layers, reshape_transform):
        # 将 model 参数赋值给类的属性 model，表示要分析的 PyTorch 模型
        self.model = model
        # 创建一个空列表，用于存储梯度值
        self.gradients = []
        # 创建一个空列表，用于存储激活值
        self.activations = []
        # 将 reshape_transform 参数赋值给类的属性 reshape_transform，表示要对激活值和梯度值进行的变形操作
        self.reshape_transform = reshape_transform
        # 创建一个空列表，用于存储注册的钩子函数
        self.handles = []
        # 遍历 target_layers 参数，表示要提取激活值和梯度值的中间层
        for target_layer in target_layers:
            # 对每个目标层，注册一个前向钩子函数，用于保存激活值，并将返回的句柄添加到 handles 列表中
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))
            # 为了兼容旧版本的 PyTorch，判断目标层是否有 register_full_backward_hook 方法
            if hasattr(target_layer, 'register_full_backward_hook'):
                # 如果有，注册一个全反向钩子函数，用于保存梯度值，并将返回的句柄添加到 handles 列表中
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:
                # 如果没有，注册一个反向钩子函数，用于保存梯度值，并将返回的句柄添加到 handles 列表中
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))

    # 定义一个方法，名为 save_activation，接受三个参数：module, input, output
    def save_activation(self, module, input, output):
        # 将 output 参数赋值给 activation 变量，表示目标层的输出（即激活值）
        activation = output
        # 判断是否有 reshape_transform 属性（即是否需要对激活值进行变形操作）
        if self.reshape_transform is not None:
            # 如果有，调用 reshape_transform 方法对 activation 变量进行变形操作，并重新赋值给 activation 变量
            activation = self.reshape_transform(activation)
        # 将 activation 变量从 GPU 移动到 CPU，并从计算图中分离出来（即不需要计算梯度），然后添加到 activations 列表中
        self.activations.append(activation.cpu().detach())
    # 定义一个方法，名为 save_gradient，接受三个参数：module, grad_input, grad_output
    def save_gradient(self, module, grad_input, grad_output):
        # 由于梯度是按照反向传播的顺序计算的，所以我们只需要取 grad_output 参数的第一个元素（即目标层输出对应的梯度），并赋值给 grad 变量
        grad = grad_output[0]
        # 判断是否有 reshape_transform 属性（即是否需要对梯度值进行变形操作）
        if self.reshape_transform is not None:
            # 如果有，调用 reshape_transform 方法对 grad 变量进行变形操作，并重新赋值给 grad 变量
            grad = self.reshape_transform(grad)
        # 将 grad 变量从 GPU 移动到 CPU，并从计算图中分离出来（即不需要计算梯度），然后添加到 gradients 列表的开头（注意这里是在列表前面插入元素，而不是在后面追加元素）
        self.gradients = [grad.cpu().detach()] + self.gradients
    # 定义一个特殊方法，名为 __call__，接受一个参数：x
    def __call__(self, x):
        # 这个方法使得类的实例可以像函数一样被调用，例如：activations_and_gradients = ActivationsAndGradients(model, target_layers, reshape_transform); activations_and_gradients(x)
        # 在调用类的实例之前，先清空 gradients 列表和 activations 列表，以便存储新的梯度值和激活值
        self.gradients = []
        self.activations = []
        # 调用 model 属性（即 PyTorch 模型）的 __call__ 方法，将 x 参数（即输入数据）作为输入，得到模型的输出，并返回
        return self.model(x)
    # 定义一个方法，名为 release，不接受任何参数
    def release(self):
        # 这个方法用于释放注册的钩子函数，以避免内存泄漏或影响其他计算
        # 遍历 handles 列表中的每个句柄
        for handle in self.handles:
            # 调用句柄的 remove 方法，移除对应的钩子函数
            handle.remove()