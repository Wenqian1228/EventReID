import torch
from  torch import nn
import  torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
import os
from models.backbone.resnet import *
import sys
import random
from feature_visualizer import FeatureVisualizer
V = FeatureVisualizer(
    cmap_type='jet',
    reduce_type='mean',
)


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}

##################################################################################
##################################################################################
##################################################################################




##########
# Basic layers
##########
class ConvLayer(nn.Module):
    """Convolution layer (conv + bn + relu)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        IN=False
    ):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=groups
        )
        if IN:
            self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1(nn.Module):
    """1x1 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=stride,
            padding=0,
            bias=False,
            groups=groups
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1Linear(nn.Module):
    """1x1 convolution + bn (w/o non-linearity)."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv1x1Linear, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1, stride=stride, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Conv3x3(nn.Module):
    """3x3 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            stride=stride,
            padding=1,
            bias=False,
            groups=groups
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LightConv3x3(nn.Module):
    """Lightweight 3x3 convolution.

    1x1 (linear) + dw 3x3 (nonlinear).
    """

    def __init__(self, in_channels, out_channels):
        super(LightConv3x3, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 1, stride=1, padding=0, bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            bias=False,
            groups=out_channels
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


##########
# Building blocks for omni-scale feature learning
##########
class ChannelGate(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(
        self,
        in_channels,
        num_gates=None,
        return_gates=False,
        gate_activation='sigmoid',
        reduction=16,
        layer_norm=False
    ):
        super(ChannelGate, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            in_channels,
            in_channels // reduction,
            kernel_size=1,
            bias=True,
            padding=0
        )
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((in_channels // reduction, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            in_channels // reduction,
            num_gates,
            kernel_size=1,
            bias=True,
            padding=0
        )
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError(
                "Unknown gate activation: {}".format(gate_activation)
            )

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x


class OSBlock(nn.Module):
    """Omni-scale feature learning block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        IN=False,
        bottleneck_reduction=4,
        **kwargs
    ):
        super(OSBlock, self).__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
        self.IN = None
        if IN:
            self.IN = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):

        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)




        identity = x
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) + self.gate(x2d)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        if self.IN is not None:
            out = self.IN(out)


        out = F.relu(out)
        out = out.view(b, -1, c, h, w)

        return out


##################################################################################
##################################################################################
##################################################################################

def init_pretrained_weight(model, model_url):
    """Initializes model with pretrained weight

    Layers that don't match with pretrained layers in name or size are kept unchanged
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weight_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class OSNet_visual(nn.Module):

    def __init__(self, num_classes, model_name, pretrain_choice, seq_len=8):
        super(OSNet_visual, self).__init__()

        self.in_planes = 2048
        self.base = ResNet()

        # if pretrain_choice == 'imagenet':
        #     init_pretrained_weight(self.base, model_urls[model_name])
        #     print('Loading pretrained ImageNet model ......')

        self.seq_len = seq_len
        self.num_classes = num_classes
        self.plances = 1024
        self.mid_channel = 256

        self.avg_2d = nn.AdaptiveAvgPool2d((1, 1))
        self.avg_3d = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.down_channel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_planes, out_channels=self.plances, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.plances),
            self.relu
        )

        t = seq_len

        self.bottleneck = nn.ModuleList([nn.BatchNorm1d(self.plances) for _  in range(3)])
        self.classifier = nn.ModuleList([nn.Linear(self.plances, num_classes) for _ in range(3)])

        self.bottleneck[0].bias.requires_grad_(False)
        self.bottleneck[1].bias.requires_grad_(False)
        self.bottleneck[2].bias.requires_grad_(False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weight_init_classifier)




#############################################################################
#############################################################################
#############################################################################

        self.os_conv_layer1 = OSBlock(self.plances,self.plances)
        self.os_conv_layer2 = OSBlock(self.plances,self.plances)
        self.os_conv_layer3 = OSBlock(self.plances,self.plances)
        self.initial_seed = 1

    def _make_layer(
        self,
        block,
        layer,
        in_channels,
        out_channels,
        reduce_spatial_size,
        IN=False
    ):
        layers = []

        layers.append(block(in_channels, out_channels, IN=IN))
        for i in range(1, layer):
            layers.append(block(out_channels, out_channels, IN=IN))

        if reduce_spatial_size:
            layers.append(
                nn.Sequential(
                    Conv1x1(out_channels, out_channels),
                    nn.AvgPool2d(2, stride=2)
                )
            )

        return nn.Sequential(*layers)


    def forward(self, x, pids=None, camid=None):    # x=[16,8,3,256,128]

        b, t, c, w, h = x.size()    # [16,8,3,256,128] [b, t, c, w, h]
        x = x.contiguous().view(b * t, c, w, h)  # x=[128,3,256,128]
        # 调用里面的模块，然后提取特征
        
        # feature_map1 =  self.base.conv1(x)
        # feature_map2 = self.base.conv2(feature_map1)
        img = x
        feat_map = self.base(x)  # (b * t, c, 16, 8)  feat_map= torch.Size([128, 2048, 16, 8])      
        # print('feat_map=',feat_map.shape)

        f1 = self.base.conv1(x)
        f1_1 = f1
        f1 = self.base.bn1(f1)
        f2 = self.base.relu(f1)
        f2 = self.base.maxpool(f2)
        f3 = self.base.layer1(f2)
        f4 = self.base.layer2(f3)
        f5 = self.base.layer3(f4)
        f6 = self.base.layer4(f5)
        # _rand_seed = random.randint(1, 100)
        
        original_path = 'visual_prid_event'
        ########################################################################
        ########################################################################
        ########################################################################
        _root_path = '{}/event_f1_1_no_erase'.format(original_path)
        if not os.path.exists(_root_path):
            os.makedirs(_root_path)
        _save_feat_path = '{}/000{}_feat.png'.format(_root_path,self.initial_seed)
        V.save_feature(f1_1, save_path=_save_feat_path)
        _save_both_path = '{}/000{}_both.png'.format(_root_path,self.initial_seed)
        V.save_both(img, f1_1, _save_both_path, recover=False)
        _save_person_path = '{}/000{}.png'.format(_root_path,self.initial_seed)
        V.save_image_single(img, save_path=_save_person_path, recover=False)
        ########################################################################
        ########################################################################
        ########################################################################
        _root_path = '{}/event_f1_no_erase'.format(original_path)
        if not os.path.exists(_root_path):
            os.makedirs(_root_path)
        _save_feat_path = '{}/000{}_feat.png'.format(_root_path,self.initial_seed)
        V.save_feature(f1, save_path=_save_feat_path)
        _save_both_path = '{}/000{}_both.png'.format(_root_path,self.initial_seed)
        V.save_both(img, f1, _save_both_path, recover=False)
        ########################################################################
        ########################################################################
        ########################################################################
        _root_path = '{}/event_f2_no_erase'.format(original_path)
        if not os.path.exists(_root_path):
            os.makedirs(_root_path)
        _save_feat_path = '{}/000{}_feat.png'.format(_root_path,self.initial_seed)
        V.save_feature(f2, save_path=_save_feat_path)
        _save_both_path = '{}/000{}_both.png'.format(_root_path,self.initial_seed)
        V.save_both(img, f2, _save_both_path, recover=False)
        ########################################################################
        ########################################################################
        ########################################################################
        _root_path = '{}/event_f3_no_erase'.format(original_path)
        if not os.path.exists(_root_path):
            os.makedirs(_root_path)
        _save_feat_path = '{}/000{}_feat.png'.format(_root_path,self.initial_seed)
        V.save_feature(f3, save_path=_save_feat_path)
        _save_both_path = '{}/000{}_both.png'.format(_root_path,self.initial_seed)
        V.save_both(img, f3, _save_both_path, recover=False)
        ########################################################################
        ########################################################################
        ########################################################################
        _root_path = '{}/event_f4_no_erase'.format(original_path)
        if not os.path.exists(_root_path):
            os.makedirs(_root_path)
        _save_feat_path = '{}/000{}_feat.png'.format(_root_path,self.initial_seed)
        V.save_feature(f4, save_path=_save_feat_path)
        _save_both_path = '{}/000{}_both.png'.format(_root_path,self.initial_seed)
        V.save_both(img, f4, _save_both_path, recover=False)
        ########################################################################
        ########################################################################
        ########################################################################
        _root_path = '{}/event_f6_no_erase'.format(original_path)
        if not os.path.exists(_root_path):
            os.makedirs(_root_path)
        _save_feat_path = '{}/000{}_feat.png'.format(_root_path,self.initial_seed)
        V.save_feature(f6, save_path=_save_feat_path)
        _save_both_path = '{}/000{}_both.png'.format(_root_path,self.initial_seed)
        V.save_both(img, f6, _save_both_path, recover=False)
        
        
        ########################################################################
        ########################################################################
        ########################################################################
        _root_path = '{}/event_f5_no_erase'.format(original_path)
        if not os.path.exists(_root_path):
            os.makedirs(_root_path)        

        _save_feat_path = '{}/000{}_feat.png'.format(_root_path,self.initial_seed)

        print('_save_feat_path=',_save_feat_path)
        V.save_feature(f5, save_path=_save_feat_path)
        _save_both_path = '{}/000{}_both.png'.format(_root_path,self.initial_seed)
        V.save_both(img, f1_1, _save_both_path, recover=False)
        self.initial_seed = self.initial_seed + 1
        if self.initial_seed == 200:
            sys.exit(0)
        
        ########################################################################
        ########################################################################
        ########################################################################

        
        # _root_path = 'visual/normal_img'
        # if not os.path.exists(_root_path):
        #     os.makedirs(_root_path)        

        # # _save_feat_path = '{}/000{}_feat.png'.format(_root_path,self.initial_seed)

        # # print('_save_feat_path=',_save_feat_path)
        # # V.save_feature(f5, save_path=_save_feat_path)
        # # _save_both_path = '{}/000{}_both.png'.format(_root_path,self.initial_seed)
        # # V.save_both(img, f1_1, _save_both_path, recover=False)
        # _save_person_path = '{}/000{}_person.png'.format(_root_path,self.initial_seed)
        # V.save_image_single(img, save_path=_save_person_path, recover=False)
        # print('_save_person_path=',_save_person_path)
        # self.initial_seed = self.initial_seed + 1
        # if self.initial_seed == 100:
        #     sys.exit(0)




        w = feat_map.size(2)
        h = feat_map.size(3)

        feat_map = self.down_channel(feat_map)
        feat_map = feat_map.view(b, t, -1, w, h)    # [4, 8, 1024, 16, 8]
        feature_list = []
        list = []
        # print('feat_map=',feat_map.shape)
            
        feat_map_1 = self.os_conv_layer1(feat_map)
        feature_1 = torch.mean(feat_map_1, 1)
        feature1 = self.avg_2d(feature_1).view(b, -1)   # [4, 1024]
        feature_list.append(feature1)
        list.append(feature1)
        # print('feature1=',feature1.shape)

        feat_map_2 = self.os_conv_layer1(feat_map_1)
        feature_2 = torch.mean(feat_map_2, 1)           # # [4, 1024]
        feature_2 = self.avg_2d(feature_2).view(b, -1)
        list.append(feature_2)
        # print('feature_2=',feature_2.shape)

        feature2 = torch.stack(list, 1)
        feature2 = torch.mean(feature2, 1)              # [4, 1024]
        feature_list.append(feature2)
        # print('feature2=',feature2.shape)

        feat_map_3 = self.os_conv_layer1(feat_map_2)
        feature_3 = torch.mean(feat_map_3, 1)
        feature_3 = self.avg_2d(feature_3).view(b, -1)  # [4, 1024]
        list.append(feature_3)
        # print('feature_3=',feature_3.shape)

        feature3 = torch.stack(list, 1)
        feature3 = torch.mean(feature3, 1)          # [4, 1024]
        feature_list.append(feature3)
        # print('feature3=',feature3.shape)
        # sys.exit()

        BN_feature_list = []
        for i in range(len(feature_list)):
            BN_feature_list.append(self.bottleneck[i](feature_list[i]))
        torch.cuda.empty_cache()

        cls_score = []
        for i in range(len(BN_feature_list)):
            cls_score.append(self.classifier[i](BN_feature_list[i]))

        if self.training:
            return cls_score, BN_feature_list
        else:
            return BN_feature_list[2], pids, camid