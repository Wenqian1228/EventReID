import torch
from  torch import nn
import  torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
from models.backbone.resnet import *
from models.STAM import STAM
import sys
import os
from feature_visualizer2 import FeatureVisualizer
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
#############################################################################


class DeformConv2D(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, bias=None):
        super(DeformConv2D, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv_kernel = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

    def forward(self, x, offset):
        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        # Change offset's order from [x1, x2, ..., y1, y2, ...] to [x1, y1, x2, y2, ...]
        # Codes below are written to make sure same results of MXNet implementation.
        # You can remove them, and it won't influence the module's performance.
        offsets_index = Variable(torch.cat([torch.arange(0, 2*N, 2), torch.arange(1, 2*N+1, 2)]), requires_grad=False).type_as(x).long()
        offsets_index = offsets_index.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1).expand(*offset.size())
        offset = torch.gather(offset, dim=1, index=offsets_index)
        # ------------------------------------------------------------------------

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = Variable(p.data, requires_grad=False).floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], -1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], -1)

        # (b, h, w, N)
        mask = torch.cat([p[..., :N].lt(self.padding)+p[..., :N].gt(x.size(2)-1-self.padding),
                          p[..., N:].lt(self.padding)+p[..., N:].gt(x.size(3)-1-self.padding)], dim=-1).type_as(p)
        mask = mask.detach()
        floor_p = p - (p - torch.floor(p))
        p = p*(1-mask) + floor_p*mask
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv_kernel(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = np.meshgrid(range(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
                          range(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1), indexing='ij')
        # (2N, 1)
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten()))
        p_n = np.reshape(p_n, (1, 2*N, 1, 1))
        p_n = Variable(torch.from_numpy(p_n).type(dtype), requires_grad=False)

        return p_n

    @staticmethod
    def _get_p_0(h, w, N, dtype):
        p_0_x, p_0_y = np.meshgrid(range(1, h+1), range(1, w+1), indexing='ij')
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w).repeat(N, axis=1)
        p_0 = np.concatenate((p_0_x, p_0_y), axis=1)
        p_0 = Variable(torch.from_numpy(p_0).type(dtype), requires_grad=False)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset

##########################################################################
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

class OSNet_img_event_visual(nn.Module):

    def __init__(self, num_classes, model_name, pretrain_choice, seq_len=8):
        super(OSNet_img_event_visual, self).__init__()

        self.in_planes = 2048
        self.base = ResNet()

        if pretrain_choice == 'imagenet':
            init_pretrained_weight(self.base, model_urls[model_name])
            print('Loading pretrained ImageNet model ......')

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
        self.layer1 = STAM(inplanes=self.plances, mid_planes=self.mid_channel, seq_len=t / 2, num= '1')

        t = t / 2
        self.layer2 = STAM(inplanes=self.plances, mid_planes=self.mid_channel, seq_len=t / 2, num= '2')

        t = t / 2
        self.layer3 = STAM(inplanes=self.plances, mid_planes=self.mid_channel, seq_len=t / 2, num= '3')


        self.bottleneck = nn.ModuleList([nn.BatchNorm1d(self.plances) for _  in range(3)])
        self.classifier = nn.ModuleList([nn.Linear(self.plances, num_classes) for _ in range(3)])

        self.bottleneck[0].bias.requires_grad_(False)
        self.bottleneck[1].bias.requires_grad_(False)
        self.bottleneck[2].bias.requires_grad_(False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weight_init_classifier)


        self.offsets = nn.Conv2d(3, 18, kernel_size=3, padding=1)
        self.deformconv = DeformConv2D(3, 3, kernel_size=3, padding=1)


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
        # print('x_origin=',x.shape)  #x = [SEQS_PER_BATCH,]
        _, _t, _, _, _ = x.size()    # [16,8,3,256,128] [b, t, c, w, h]
        
        event = x[:,_t//2:,:,:,:]
        b, t, c, w, h = event.size()
        event = event.contiguous().view(b * t, c, w, h)
        # offsets = self.offsets(event)
        # event = F.relu(self.deformconv(event, offsets))     # x = [64,3,256,128]

        # event_feat_map = self.base(event)
        img = event
        f1 = self.base.conv1(event)
        # print('f1=',f1.shape)
        f1_1 = f1
        f1 = self.base.bn1(f1)
        f2 = self.base.relu(f1)
        f2 = self.base.maxpool(f2)
        f3 = self.base.layer1(f2)
        f4 = self.base.layer2(f3)
        f5 = self.base.layer3(f4)
        event_feat_map = self.base.layer4(f5)



        # _root_path = 'visual/f1_1_no_erase_no_normalize'
        # if not os.path.exists(_root_path):
        #     os.makedirs(_root_path)
        # _save_both_path = '{}/000{}_both.png'.format(_root_path,self.initial_seed)
        # V.save_both(img, f1_1, _save_both_path, recover=False)
        # ########################################################################
        # ########################################################################
        # ########################################################################
        # x = x[:,0:_t//2,:,:,:]
        # b, t, c, w, h = x.size()    # [16,8,3,256,128] [b, t, c, w, h]
        # x = x.contiguous().view(b * t, c, w, h)  # x=[128,3,256,128]

        # img = x
        # f1 = self.base.conv1(x)
        # f1_1 = f1
        # f1 = self.base.bn1(f1)
        # f2 = self.base.relu(f1)
        # f2 = self.base.maxpool(f2)
        # f3 = self.base.layer1(f2)
        # f4 = self.base.layer2(f3)
        # f5 = self.base.layer3(f4)
        # feat_map = self.base.layer4(f5)


        # ########################################################################
        # ########################################################################
        # ########################################################################
        # _root_path = 'visual/event_f1_1_no_erase_no_normalize'
        # if not os.path.exists(_root_path):
        #     os.makedirs(_root_path)
        # # _save_both_path = '{}/000{}_both.png'.format(_root_path,self.initial_seed)
        # # V.save_both(img, f1_1, _save_both_path, recover=False)
        # # print('_save_both_path=',_save_both_path)


        # _save_path='{}/000{}.png'.format(_root_path,self.initial_seed)
        # _save_person_path = '{}/000{}_person.png'.format(_root_path,self.initial_seed)

        # print('_save_both_path=',_save_both_path)
        # V.save_feature(f1, save_path=_save_path)
        # V.save_image_single(img, save_path=_save_person_path, recover=False)
        # self.initial_seed = self.initial_seed + 1
        # if self.initial_seed == 100:
        #     sys.exit(0)

        
        ########################################################################
        ########################################################################
        ########################################################################
        # _root_path = 'visual/event_f1_1_no_erase'
        # if not os.path.exists(_root_path):
        #     os.makedirs(_root_path)
        # _save_path='{}/000{}_0.png'.format(_root_path,self.initial_seed)
        # _save_person_path = '{}/000{}_person_0.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f1_1[0], save_path=_save_path)
        # V.save_image_single(img[0], save_path=_save_person_path, recover=False)

        # _save_path='{}/000{}_1.png'.format(_root_path,self.initial_seed)
        # _save_person_path = '{}/000{}_person_1.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f1_1[1], save_path=_save_path)
        # V.save_image_single(img[1], save_path=_save_person_path, recover=False)

        # _save_path='{}/000{}_2.png'.format(_root_path,self.initial_seed)
        # _save_person_path = '{}/000{}_person_2.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f1_1[2], save_path=_save_path)
        # V.save_image_single(img[2], save_path=_save_person_path, recover=False)

        # _save_path='{}/000{}_3.png'.format(_root_path,self.initial_seed)
        # _save_person_path = '{}/000{}_person_3.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f1_1[3], save_path=_save_path)
        # V.save_image_single(img[3], save_path=_save_person_path, recover=False)

        # _save_path='{}/000{}_4.png'.format(_root_path,self.initial_seed)
        # _save_person_path = '{}/000{}_person_4.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f1_1[4], save_path=_save_path)
        # V.save_image_single(img[4], save_path=_save_person_path, recover=False)

        # _save_path='{}/000{}_5.png'.format(_root_path,self.initial_seed)
        # _save_person_path = '{}/000{}_person_5.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f1_1[5], save_path=_save_path)
        # V.save_image_single(img[5], save_path=_save_person_path, recover=False)

        # _save_path='{}/000{}_6.png'.format(_root_path,self.initial_seed)
        # _save_person_path = '{}/000{}_person_6.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f1_1[6], save_path=_save_path)
        # V.save_image_single(img[6], save_path=_save_person_path, recover=False)

        # _save_path='{}/000{}_7.png'.format(_root_path,self.initial_seed)
        # _save_person_path = '{}/000{}_person_7.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f1_1[7], save_path=_save_path)
        # V.save_image_single(img[7], save_path=_save_person_path, recover=False)

        # _save_path='{}/000{}_8.png'.format(_root_path,self.initial_seed)
        # _save_person_path = '{}/000{}_person_8.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f1_1[8], save_path=_save_path)
        # V.save_image_single(img[8], save_path=_save_person_path, recover=False)


        ########################################################################
        ########################################################################
        ########################################################################
        # _root_path = 'visual/event_f1_no_erase'
        # if not os.path.exists(_root_path):
        #     os.makedirs(_root_path)
        # _save_path='{}/000{}_0.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f1[0], save_path=_save_path)
        # _save_person_path = '{}/000{}_person_0.png'.format(_root_path,self.initial_seed)
        # V.save_image_single(img[0], save_path=_save_person_path, recover=False)

        # _save_path='{}/000{}_1.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f1[1], save_path=_save_path)
        # _save_person_path = '{}/000{}_person_1.png'.format(_root_path,self.initial_seed)
        # V.save_image_single(img[1], save_path=_save_person_path, recover=False)

        # _save_path='{}/000{}_2.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f1[2], save_path=_save_path)
        # _save_person_path = '{}/000{}_person_2.png'.format(_root_path,self.initial_seed)
        # V.save_image_single(img[2], save_path=_save_person_path, recover=False)

        # _save_path='{}/000{}_3.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f1[3], save_path=_save_path)
        # _save_person_path = '{}/000{}_person_3.png'.format(_root_path,self.initial_seed)
        # V.save_image_single(img[3], save_path=_save_person_path, recover=False)

        # _save_path='{}/000{}_4.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f1[4], save_path=_save_path)
        # _save_person_path = '{}/000{}_person_4.png'.format(_root_path,self.initial_seed)
        # V.save_image_single(img[4], save_path=_save_person_path, recover=False)

        # _save_path='{}/000{}_5.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f1[5], save_path=_save_path)
        # _save_person_path = '{}/000{}_person_5.png'.format(_root_path,self.initial_seed)
        # V.save_image_single(img[5], save_path=_save_person_path, recover=False)

        # _save_path='{}/000{}_6.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f1[6], save_path=_save_path)
        # _save_person_path = '{}/000{}_person_6.png'.format(_root_path,self.initial_seed)
        # V.save_image_single(img[6], save_path=_save_person_path, recover=False)

        # _save_path='{}/000{}_7.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f1[7], save_path=_save_path)
        # _save_person_path = '{}/000{}_person_7.png'.format(_root_path,self.initial_seed)
        # V.save_image_single(img[7], save_path=_save_person_path, recover=False)

        # _save_path='{}/000{}_8.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f1[8], save_path=_save_path)
        # _save_person_path = '{}/000{}_person_8.png'.format(_root_path,self.initial_seed)
        # V.save_image_single(img[8], save_path=_save_person_path, recover=False)

        # _save_path='{}/000{}_9.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f1[9], save_path=_save_path)
        # _save_person_path = '{}/000{}_person_9.png'.format(_root_path,self.initial_seed)
        # V.save_image_single(img[9], save_path=_save_person_path, recover=False)

        # _save_path='{}/000{}_10.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f1[10], save_path=_save_path)
        # _save_person_path = '{}/000{}_person_10.png'.format(_root_path,self.initial_seed)
        # V.save_image_single(img[10], save_path=_save_person_path, recover=False)

        # _save_path='{}/000{}_11.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f1[11], save_path=_save_path)
        # _save_person_path = '{}/000{}_person_11.png'.format(_root_path,self.initial_seed)
        # V.save_image_single(img[11], save_path=_save_person_path, recover=False)

        # _save_path='{}/000{}_12.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f1[12], save_path=_save_path)
        # _save_person_path = '{}/000{}_person_12.png'.format(_root_path,self.initial_seed)
        # V.save_image_single(img[12], save_path=_save_person_path, recover=False)

        # _save_path='{}/000{}_13.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f1[13], save_path=_save_path)
        # _save_person_path = '{}/000{}_person_13.png'.format(_root_path,self.initial_seed)
        # V.save_image_single(img[13], save_path=_save_person_path, recover=False)

        # _save_path='{}/000{}_14.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f1[14], save_path=_save_path)
        # _save_person_path = '{}/000{}_person_14.png'.format(_root_path,self.initial_seed)
        # V.save_image_single(img[14], save_path=_save_person_path, recover=False)

        # _save_path='{}/000{}_15.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f1[15], save_path=_save_path)
        # _save_person_path = '{}/000{}_person_15.png'.format(_root_path,self.initial_seed)
        # V.save_image_single(img[15], save_path=_save_person_path, recover=False)

        # ########################################################################
        # ########################################################################
        # ########################################################################
        # _root_path = 'visual/f2_no_erase'
        # if not os.path.exists(_root_path):
        #     os.makedirs(_root_path)
        # _save_path='{}/000{}.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f2, save_path=_save_path)
        # _save_both_path = '{}/000{}_both.png'.format(_root_path,self.initial_seed)
        # V.save_both(img, f2, _save_both_path, recover=False)
        # ########################################################################
        # ########################################################################
        # ########################################################################
        # _root_path = 'visual/f3_no_erase'
        # if not os.path.exists(_root_path):
        #     os.makedirs(_root_path)
        # _save_path='{}/000{}.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f3, save_path=_save_path)
        # _save_both_path = '{}/000{}_both.png'.format(_root_path,self.initial_seed)
        # V.save_both(img, f3, _save_both_path, recover=False)
        # ########################################################################
        # ########################################################################
        # ########################################################################
        # _root_path = 'visual/f4_no_erase'
        # if not os.path.exists(_root_path):
        #     os.makedirs(_root_path)
        # _save_path='{}/000{}.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f4, save_path=_save_path)
        # _save_both_path = '{}/000{}_both.png'.format(_root_path,self.initial_seed)
        # V.save_both(img, f4, _save_both_path, recover=False)
        # ########################################################################
        # ########################################################################
        # ########################################################################
        # _root_path = 'visual/f6_no_erase'
        # if not os.path.exists(_root_path):
        #     os.makedirs(_root_path)
        # _save_path='{}/000{}.png'.format(_root_path,self.initial_seed)
        # V.save_feature(event_feat_map, save_path=_save_path)
        # _save_both_path = '{}/000{}_both.png'.format(_root_path,self.initial_seed)
        # V.save_both(img, event_feat_map, _save_both_path, recover=False)
        # ########################################################################
        # ########################################################################
        # ########################################################################
        # _root_path = 'visual/f5_no_erase'
        # if not os.path.exists(_root_path):
        #     os.makedirs(_root_path)        
        # _save_path='{}/000{}.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f5, save_path=_save_path)
        # _save_both_path = '{}/000{}_both.png'.format(_root_path,self.initial_seed)
        # V.save_both(img, f5, _save_both_path, recover=False)
        ##############################################################################
        ##############################################################################
        
        x = x[:,0:_t//2,:,:,:]
        b, t, c, w, h = x.size()    # [16,8,3,256,128] [b, t, c, w, h]
        x = x.contiguous().view(b * t, c, w, h)  # x=[128,3,256,128]
        # 调用里面的模块，然后提取特征
        
        # feat_map = self.base(x)   
        img = x
        f1 = self.base.conv1(x)     # f1_1 = [16,64,64,32]
        f1_1 = f1
        f1 = self.base.bn1(f1)
        f2 = self.base.relu(f1)
        f2 = self.base.maxpool(f2)
        f3 = self.base.layer1(f2)
        f4 = self.base.layer2(f3)
        f5 = self.base.layer3(f4)
        feat_map = self.base.layer4(f5)
        ########################################################################
        ########################################################################
        ########################################################################
        # _root_path = 'visual/event_f1_1_no_erase'
        # if not os.path.exists(_root_path):
        #     os.makedirs(_root_path)
        
        # _save_path='{}/000{}_0.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f1_1[0], save_path=_save_path)
        # _save_path='{}/000{}_1.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f1_1[1], save_path=_save_path)
        # _save_path='{}/000{}_2.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f1_1[2], save_path=_save_path)
        # _save_path='{}/000{}_3.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f1_1[3], save_path=_save_path)
        # _save_path='{}/000{}_4.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f1_1[4], save_path=_save_path)
        # _save_path='{}/000{}_5.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f1_1[5], save_path=_save_path)
        # _save_path='{}/000{}_6.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f1_1[6], save_path=_save_path)
        # _save_path='{}/000{}_7.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f1_1[7], save_path=_save_path)
        # ###############################################################
        # _save_person_path0 = '{}/000{}_person0.png'.format(_root_path,self.initial_seed)
        # V.save_image_single(img[0], save_path=_save_person_path0, recover=False)
        # _save_person_path0 = '{}/000{}_person1.png'.format(_root_path,self.initial_seed)
        # V.save_image_single(img[1], save_path=_save_person_path0, recover=False)
        # _save_person_path0 = '{}/000{}_person2.png'.format(_root_path,self.initial_seed)
        # V.save_image_single(img[2], save_path=_save_person_path0, recover=False)
        # _save_person_path0 = '{}/000{}_person3.png'.format(_root_path,self.initial_seed)
        # V.save_image_single(img[3], save_path=_save_person_path0, recover=False)
        # _save_person_path0 = '{}/000{}_person4.png'.format(_root_path,self.initial_seed)
        # V.save_image_single(img[4], save_path=_save_person_path0, recover=False)
        # _save_person_path0 = '{}/000{}_person5.png'.format(_root_path,self.initial_seed)
        # V.save_image_single(img[5], save_path=_save_person_path0, recover=False)
        # _save_person_path0 = '{}/000{}_person6.png'.format(_root_path,self.initial_seed)
        # V.save_image_single(img[6], save_path=_save_person_path0, recover=False)
        # _save_person_path0 = '{}/000{}_person7.png'.format(_root_path,self.initial_seed)
        # V.save_image_single(img[7], save_path=_save_person_path0, recover=False)
        # ##############################################################
        # _save_person_path0 = '{}/000{}_person8.png'.format(_root_path,self.initial_seed)
        # V.save_image_single(img[8], save_path=_save_person_path0, recover=False)
        # _save_person_path0 = '{}/000{}_person9.png'.format(_root_path,self.initial_seed)
        # V.save_image_single(img[9], save_path=_save_person_path0, recover=False)
        # _save_person_path0 = '{}/000{}_person10.png'.format(_root_path,self.initial_seed)
        # V.save_image_single(img[10], save_path=_save_person_path0, recover=False)
        # _save_person_path0 = '{}/000{}_person11.png'.format(_root_path,self.initial_seed)
        # V.save_image_single(img[11], save_path=_save_person_path0, recover=False)
        # _save_person_path0 = '{}/000{}_person12.png'.format(_root_path,self.initial_seed)
        # V.save_image_single(img[12], save_path=_save_person_path0, recover=False)
        # _save_person_path0 = '{}/000{}_person13.png'.format(_root_path,self.initial_seed)
        # V.save_image_single(img[13], save_path=_save_person_path0, recover=False)
        # _save_person_path0 = '{}/000{}_person14.png'.format(_root_path,self.initial_seed)
        # V.save_image_single(img[14], save_path=_save_person_path0, recover=False)
        # _save_person_path0 = '{}/000{}_person15.png'.format(_root_path,self.initial_seed)
        # V.save_image_single(img[15], save_path=_save_person_path0, recover=False)
        


        #####################################################################
        #####################################################################
        #####################################################################

        _root_path = 'visual/f1_1_no_erase'
        if not os.path.exists(_root_path):
            os.makedirs(_root_path)

        _save_path='{}/000{}_0.png'.format(_root_path,self.initial_seed)
        V.save_feature(f1_1[0], save_path=_save_path)
        _save_both_path = '{}/000{}_both_0.png'.format(_root_path,self.initial_seed)
        V.save_both(img[0], f1_1[0], _save_both_path, recover=False)
        _save_person_path = '{}/000{}_person_0.png'.format(_root_path,self.initial_seed)
        V.save_image_single(img[0], save_path=_save_person_path, recover=False)
        # print('img[0]=',img[0])

        _save_path='{}/000{}_1.png'.format(_root_path,self.initial_seed)
        V.save_feature(f1_1[1], save_path=_save_path)
        _save_both_path = '{}/000{}_both_1.png'.format(_root_path,self.initial_seed)
        V.save_both(img[1], f1_1[1], _save_both_path, recover=False)
        _save_person_path = '{}/000{}_person_0.png'.format(_root_path,self.initial_seed)
        V.save_image_single(img[1], save_path=_save_person_path, recover=False)

        _save_path='{}/000{}_2.png'.format(_root_path,self.initial_seed)
        V.save_feature(f1_1[2], save_path=_save_path)
        _save_both_path = '{}/000{}_both_2.png'.format(_root_path,self.initial_seed)
        V.save_both(img[2], f1_1[2], _save_both_path, recover=False)
        _save_person_path = '{}/000{}_person_2.png'.format(_root_path,self.initial_seed)
        V.save_image_single(img[2], save_path=_save_person_path, recover=False)

        _save_path='{}/000{}_3.png'.format(_root_path,self.initial_seed)
        V.save_feature(f1_1[3], save_path=_save_path)
        _save_both_path = '{}/000{}_both_3.png'.format(_root_path,self.initial_seed)
        V.save_both(img[3], f1_1[3], _save_both_path, recover=False)
        _save_person_path = '{}/000{}_person_3.png'.format(_root_path,self.initial_seed)
        V.save_image_single(img[3], save_path=_save_person_path, recover=False)

        _save_path='{}/000{}_4.png'.format(_root_path,self.initial_seed)
        V.save_feature(f1_1[4], save_path=_save_path)
        _save_both_path = '{}/000{}_both_4.png'.format(_root_path,self.initial_seed)
        V.save_both(img[4], f1_1[4], _save_both_path, recover=False)
        _save_person_path = '{}/000{}_person_4.png'.format(_root_path,self.initial_seed)
        V.save_image_single(img[4], save_path=_save_person_path, recover=False)

        _save_path='{}/000{}_5.png'.format(_root_path,self.initial_seed)
        V.save_feature(f1_1[5], save_path=_save_path)
        _save_both_path = '{}/000{}_both_5.png'.format(_root_path,self.initial_seed)
        V.save_both(img[5], f1_1[5], _save_both_path, recover=False)
        _save_person_path = '{}/000{}_person_5.png'.format(_root_path,self.initial_seed)
        V.save_image_single(img[5], save_path=_save_person_path, recover=False)

        _save_path='{}/000{}_6.png'.format(_root_path,self.initial_seed)
        V.save_feature(f1_1[6], save_path=_save_path)
        _save_both_path = '{}/000{}_both_6.png'.format(_root_path,self.initial_seed)
        V.save_both(img[6], f1_1[6], _save_both_path, recover=False)
        _save_person_path = '{}/000{}_person_6.png'.format(_root_path,self.initial_seed)
        V.save_image_single(img[6], save_path=_save_person_path, recover=False)

        _save_path='{}/000{}_7.png'.format(_root_path,self.initial_seed)
        V.save_feature(f1_1[7], save_path=_save_path)
        _save_both_path = '{}/000{}_both_7.png'.format(_root_path,self.initial_seed)
        V.save_both(img[7], f1_1[7], _save_both_path, recover=False)
        _save_person_path = '{}/000{}_person_7.png'.format(_root_path,self.initial_seed)
        V.save_image_single(img[7], save_path=_save_person_path, recover=False)

        _save_path='{}/000{}_8.png'.format(_root_path,self.initial_seed)
        V.save_feature(f1_1[8], save_path=_save_path)
        _save_both_path = '{}/000{}_both_8.png'.format(_root_path,self.initial_seed)
        V.save_both(img[8], f1_1[8], _save_both_path, recover=False)
        _save_person_path = '{}/000{}_person_8.png'.format(_root_path,self.initial_seed)
        V.save_image_single(img[8], save_path=_save_person_path, recover=False)

        _save_path='{}/000{}_9.png'.format(_root_path,self.initial_seed)
        V.save_feature(f1_1[9], save_path=_save_path)
        _save_both_path = '{}/000{}_both_9.png'.format(_root_path,self.initial_seed)
        V.save_both(img[9], f1_1[9], _save_both_path, recover=False)
        _save_person_path = '{}/000{}_person_9.png'.format(_root_path,self.initial_seed)
        V.save_image_single(img[9], save_path=_save_person_path, recover=False)

        _save_path='{}/000{}_10.png'.format(_root_path,self.initial_seed)
        V.save_feature(f1_1[10], save_path=_save_path)
        _save_both_path = '{}/000{}_both_10.png'.format(_root_path,self.initial_seed)
        V.save_both(img[10], f1_1[10], _save_both_path, recover=False)
        _save_person_path = '{}/000{}_person_10.png'.format(_root_path,self.initial_seed)
        V.save_image_single(img[10], save_path=_save_person_path, recover=False)

        _save_path='{}/000{}_11.png'.format(_root_path,self.initial_seed)
        V.save_feature(f1_1[11], save_path=_save_path)
        _save_both_path = '{}/000{}_both_11.png'.format(_root_path,self.initial_seed)
        V.save_both(img[11], f1_1[11], _save_both_path, recover=False)
        _save_person_path = '{}/000{}_person_11.png'.format(_root_path,self.initial_seed)
        V.save_image_single(img[11], save_path=_save_person_path, recover=False)

        _save_path='{}/000{}_12.png'.format(_root_path,self.initial_seed)
        V.save_feature(f1_1[12], save_path=_save_path)
        _save_both_path = '{}/000{}_both_12.png'.format(_root_path,self.initial_seed)
        V.save_both(img[12], f1_1[12], _save_both_path, recover=False)
        _save_person_path = '{}/000{}_person_12.png'.format(_root_path,self.initial_seed)
        V.save_image_single(img[12], save_path=_save_person_path, recover=False)

        _save_path='{}/000{}_13.png'.format(_root_path,self.initial_seed)
        V.save_feature(f1_1[13], save_path=_save_path)
        _save_both_path = '{}/000{}_both_13.png'.format(_root_path,self.initial_seed)
        V.save_both(img[13], f1_1[13], _save_both_path, recover=False)
        _save_person_path = '{}/000{}_person_13.png'.format(_root_path,self.initial_seed)
        V.save_image_single(img[13], save_path=_save_person_path, recover=False)

        _save_path='{}/000{}_14.png'.format(_root_path,self.initial_seed)
        V.save_feature(f1_1[14], save_path=_save_path)
        _save_both_path = '{}/000{}_both_14.png'.format(_root_path,self.initial_seed)
        V.save_both(img[14], f1_1[14], _save_both_path, recover=False)
        _save_person_path = '{}/000{}_person_11.png'.format(_root_path,self.initial_seed)
        V.save_image_single(img[14], save_path=_save_person_path, recover=False)

        _save_path='{}/000{}_15.png'.format(_root_path,self.initial_seed)
        V.save_feature(f1_1[15], save_path=_save_path)
        _save_both_path = '{}/000{}_both_15.png'.format(_root_path,self.initial_seed)
        V.save_both(img[15], f1_1[15], _save_both_path, recover=False)
        _save_person_path = '{}/000{}_person_15.png'.format(_root_path,self.initial_seed)
        V.save_image_single(img[15], save_path=_save_person_path, recover=False)
        # # ########################################################################
        # # ########################################################################
        # # ########################################################################
        _root_path = 'visual/f1_no_erase'
        if not os.path.exists(_root_path):
            os.makedirs(_root_path)

        _save_path='{}/000{}.png'.format(_root_path,self.initial_seed)
        V.save_feature(f1[0], save_path=_save_path)
        _save_both_path = '{}/000{}_both.png'.format(_root_path,self.initial_seed)
        V.save_both(img[0], f1[0], _save_both_path, recover=False)   # img=[16,3,128,64], f1 = [16,64,64,32]
        # # # # ########################################################################
        # # # # ########################################################################
        # # # # ########################################################################
        # _root_path = 'visual/f2_no_erase'
        # if not os.path.exists(_root_path):
        #     os.makedirs(_root_path)

        # _save_path='{}/000{}.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f2[0], save_path=_save_path)
        # _save_both_path = '{}/000{}_both.png'.format(_root_path,self.initial_seed)
        # V.save_both(img[0], f2[0], _save_both_path, recover=False)
        # # ########################################################################
        # # ########################################################################
        # # ########################################################################
        # _root_path = 'visual/event_f3_no_erase'
        # if not os.path.exists(_root_path):
        #     os.makedirs(_root_path)
        # _save_path='{}/000{}.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f3, save_path=_save_path)
        # _save_both_path = '{}/000{}_both.png'.format(_root_path,self.initial_seed)
        # V.save_both(img, f3, _save_both_path, recover=False)
        # ########################################################################
        # ########################################################################
        # ########################################################################
        # _root_path = 'visual/event_f4_no_erase'
        # if not os.path.exists(_root_path):
        #     os.makedirs(_root_path)
        # _save_path='{}/000{}.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f4, save_path=_save_path)
        # _save_both_path = '{}/000{}_both.png'.format(_root_path,self.initial_seed)
        # V.save_both(img, f4, _save_both_path, recover=False)
        # ########################################################################
        # ########################################################################
        # ########################################################################
        # _root_path = 'visual/event_f6_no_erase'
        # if not os.path.exists(_root_path):
        #     os.makedirs(_root_path)
        # _save_path='{}/000{}.png'.format(_root_path,self.initial_seed)
        # V.save_feature(feat_map, save_path=_save_path)
        # _save_both_path = '{}/000{}_both.png'.format(_root_path,self.initial_seed)
        # V.save_both(img, feat_map, _save_both_path, recover=False)
        # ########################################################################
        # ########################################################################
        # ########################################################################
        # _root_path = 'visual/event_f5_no_erase'
        # if not os.path.exists(_root_path):
        #     os.makedirs(_root_path)        
        # _save_path='{}/000{}.png'.format(_root_path,self.initial_seed)
        # V.save_feature(f5, save_path=_save_path)
        # _save_both_path = '{}/000{}_both.png'.format(_root_path,self.initial_seed)
        # V.save_both(img, f5, _save_both_path, recover=False)
        # print('_save_path=',_save_path)
        # V.save_both(img, f5, _save_both_path, recover=False)
        self.initial_seed = self.initial_seed + 1
        if self.initial_seed == 200:
            sys.exit(0)
        ########################################################################
        ########################################################################
        ########################################################################
        feat_map = feat_map + event_feat_map

        w = feat_map.size(2)
        h = feat_map.size(3)

        feat_map = self.down_channel(feat_map)
        feat_map = feat_map.view(b, t, -1, w, h)    # [4, 8, 1024, 16, 8]
        feature_list = []
        list = []

        feat_map_1 = self.os_conv_layer1(feat_map)
        feature_1 = torch.mean(feat_map_1, 1)
        feature1 = self.avg_2d(feature_1).view(b, -1)   # [4, 1024]
        feature_list.append(feature1)
        list.append(feature1)

        feat_map_2 = self.os_conv_layer1(feat_map_1)
        feature_2 = torch.mean(feat_map_2, 1)           # # [4, 1024]
        feature_2 = self.avg_2d(feature_2).view(b, -1)
        list.append(feature_2)

        feature2 = torch.stack(list, 1)
        feature2 = torch.mean(feature2, 1)              # [4, 1024]
        feature_list.append(feature2)

        feat_map_3 = self.os_conv_layer1(feat_map_2)
        feature_3 = torch.mean(feat_map_3, 1)
        feature_3 = self.avg_2d(feature_3).view(b, -1)  # [4, 1024]
        list.append(feature_3)

        feature3 = torch.stack(list, 1)
        feature3 = torch.mean(feature3, 1)          # [4, 1024]
        feature_list.append(feature3)

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