import torch
from  torch import nn
import  torch.utils.model_zoo as model_zoo
from torch.nn import functional as F

from models.backbone.resnet import *
from models.STAM import STAM
import sys
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

##################################################################################
##################################################################################
##################################################################################

class ConvRecurrent(nn.Module):
    """
    Convolutional recurrent cell (for direct comparison with spiking nets).
    """

    def __init__(self, input_size, hidden_size, kernel_size, activation=None):
        super().__init__()

        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.ff = nn.Conv2d(input_size, hidden_size, kernel_size, padding=padding)
        self.rec = nn.Conv2d(input_size, hidden_size, kernel_size, padding=padding)
        self.out = nn.Conv2d(input_size, hidden_size, kernel_size, padding=padding)
        assert activation is None, "ConvRecurrent activation cannot be set (just for compatibility)"

    def forward(self, input_, prev_state=None):
        # generate empty prev_state, if None is provided
        if prev_state is None:
            batch, _, height, width = input_.shape
            state_shape = (batch, self.hidden_size, height, width)
            prev_state = torch.zeros(*state_shape, dtype=input_.dtype, device=input_.device)

        ff = self.ff(input_)
        rec = self.rec(prev_state)
        state = torch.tanh(ff + rec)
        out = self.out(state)
        out = torch.relu(out)

        return out, state


########################################################################################

class Salient2BroadModule(nn.Module):
    def __init__(self,
                 in_dim,
                 inter_dim=None,
                 split_pos=0,
                 k=3,
                 exp_beta=5.0,
                 cpm_alpha=0.1):
        super().__init__()
        self.in_channels = in_dim
        # self.inter_channels = inter_dim or in_dim // 4
        self.inter_channels = 3
        self.pos = split_pos
        self.k = k
        self.exp_beta = exp_beta
        self.cpm_alpha = cpm_alpha

        self.kernel = nn.Sequential(
            nn.Conv3d(self.in_channels, self.k * self.k, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(self.k * self.k),
            nn.ReLU())

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)),
            nn.Conv3d(self.in_channels, self.inter_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.inter_channels, self.in_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.Sigmoid())

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _suppress(self, a, exp_beta=5.0):
        """
        :param a: (b, 1, t, h, w)
        :return:
        """
        a_sup = (a < 1).float().detach()
        a_exp = torch.exp((a-1)*a_sup*exp_beta)
        a = a_exp * a_sup + (1 - a_sup)
        return a

    def _channel_center(self, x):
        """
        :param x:  (b, c, t, h, w)
        :return:   (b, c, 1, 1, 1)
        """
        center_w_pad = torch.mean(x, dim=(2,3,4), keepdim=True)
        center_wo_pad = torch.mean(x[:,:,:,1:-1, 1:-1], dim=(2,3,4), keepdim=True)
        center = center_wo_pad/(center_w_pad + 1e-8)
        return center

    def channel_attention_layer(self, x):
        se = self.se(x)                     # (b, c, 1, 1, 1)
        center = self._channel_center(x)     # (b, c, 1, 1, 1)
        center = (center > 1).float().detach()
        return se * center

    def _forward(self, x, pos=None):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        pos = self.pos if pos is None else pos

        b, c, t, h, w = x.shape
        xf = x[:, :, :pos + 1]
        xl = x[:, :, pos + 1:]

        cal = self.channel_attention_layer(x)
        xf_se = F.relu(xf * cal)

        # k*k spatial attention
        spatial_att = self.kernel(xf_se)    # (b, k*k, tf, h, w)
        # (b, tf*hw, k*k)
        spatial_att = spatial_att.reshape(b, self.k*self.k, -1).transpose(-2, -1)
        if self.k != 1:
            spatial_att = F.normalize(spatial_att, dim=-1, p=1)
        spatial_att = F.normalize(spatial_att, dim=1, p=1)

        # obtain k*k conv kernel
        xf_reshape = xf_se.reshape(b, c, -1)
        # (b, c, 1, k, k)
        kernel = torch.matmul(xf_reshape, spatial_att)
        kernel = kernel.reshape(b, c, 1, self.k, self.k)

        # perform convolution with calculated kernel
        xl_se = F.relu(xl * cal)    # (1, b*c, tl, h, w)
        xl_reshape = xl_se.reshape(b*c, -1, h, w)

        pad = (self.k-1)//2
        xl_reshape = F.pad(xl_reshape, pad=[pad,pad,pad,pad], mode='replicate')
        xl_reshape = xl_reshape.unsqueeze(0)
        f = F.conv3d(xl_reshape, weight=kernel, bias=None, stride=1, groups=b)
        f = f / (self.k * self.k)

        # suppress operation
        f = f.reshape(b, -1, h*w)
        f = F.softmax(f, dim=-1)
        f = f.reshape(b, 1, -1, h, w).clamp_min(1e-4)

        f = 1.0 / (f * h * w)
        f = self._suppress(f, exp_beta=self.exp_beta)

        # cross propagation
        xl_res = xl * f + self.cpm_alpha * F.adaptive_avg_pool3d(xf, 1)
        xf_res = xf + self.cpm_alpha * F.adaptive_avg_pool3d((1-f)* xl, 1)/F.adaptive_avg_pool3d((1-f), 1)
        res = torch.cat([xf_res, xl_res], dim=2)

        return res

    def forward(self, x, pos=None):
        b, t, c, h, w = x.size()
        x = x.view(b, c, t, h, w)
        
        b, c, t, h, w = x.shape
        if t == 4:
            return self._forward(x, pos)
        else:
            assert t % 4 == 0
            x = x.reshape(b, c, 2, 4, h, w)
            x = x.transpose(1, 2).reshape(b*2, c, 4, h, w)
            x = self._forward(x, pos)
            x = x.reshape(b, 2, c, 4, h, w).transpose(1, 2)
            x = x.reshape(b, c, t, h, w)

        x = x.view(b, t, c, h, w)
        return x




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

class SINet(nn.Module):

    def __init__(self, num_classes, model_name, pretrain_choice, seq_len=8):
        super(SINet, self).__init__()

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



#############################################################################
#############################################################################
#############################################################################

        self.LSTM_layer1 = Salient2BroadModule(self.plances)
        self.LSTM_layer2 = Salient2BroadModule(self.plances)
        self.LSTM_layer3 = Salient2BroadModule(self.plances)









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

        feat_map = self.base(x)  # (b * t, c, 16, 8)  feat_map= torch.Size([128, 2048, 16, 8])      
 
        w = feat_map.size(2)
        h = feat_map.size(3)

        feat_map = self.down_channel(feat_map)
        feat_map = feat_map.view(b, t, -1, w, h)    # [4, 8, 1024, 16, 8]
        feature_list = []
        list = []
        # print('feat_map=',feat_map.shape)
            
        feat_map_1 = self.LSTM_layer1(feat_map)
        feature_1 = torch.mean(feat_map_1, 1)
        feature1 = self.avg_2d(feature_1).view(b, -1)   # [4, 1024]
        feature_list.append(feature1)
        list.append(feature1)
        # print('feature1=',feature1.shape)

        feat_map_2 = self.LSTM_layer2(feat_map_1)
        feature_2 = torch.mean(feat_map_2, 1)           # # [4, 1024]
        feature_2 = self.avg_2d(feature_2).view(b, -1)
        list.append(feature_2)
        # print('feature_2=',feature_2.shape)

        feature2 = torch.stack(list, 1)
        feature2 = torch.mean(feature2, 1)              # [4, 1024]
        feature_list.append(feature2)
        # print('feature2=',feature2.shape)

        feat_map_3 = self.LSTM_layer3(feat_map_2)
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