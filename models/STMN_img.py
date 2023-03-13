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

class ConvLayer(nn.Module):
    """
    Convolutional layer.
    Default: bias, ReLU, no downsampling, no batch norm.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        activation="relu",
        norm=None,
        BN_momentum=0.1,
        w_scale=None,
    ):
        super(ConvLayer, self).__init__()

        bias = False if norm == "BN" else True
        padding = kernel_size // 2
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        
        self.sigmoid = nn.Sigmoid()

        ###############################################################
        ###############################################################



        
        if w_scale is not None:
            nn.init.uniform_(self.conv2d.weight, -w_scale, w_scale)
            nn.init.zeros_(self.conv2d.bias)

        if activation is not None:
            if hasattr(torch, activation):
                self.activation = getattr(torch, activation)
            else:
                self.activation = getattr(spiking, activation)
        else:
            self.activation = None

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.conv2d(x)

        if self.norm in ["BN", "IN"]:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out
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

class RecurrentConvLayer(nn.Module):
    """
    Layer comprised of a convolution followed by a recurrent convolutional block.
    Default: bias, ReLU, no downsampling, no batch norm, ConvLSTM.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        recurrent_block_type="convlstm",
        activation_ff="relu",
        activation_rec=None,
        norm=None,
        BN_momentum=0.1,
    ):
        super(RecurrentConvLayer, self).__init__()

        assert recurrent_block_type in ["convlstm", "convgru", "convrnn"]
        self.recurrent_block_type = recurrent_block_type


        RecurrentBlock = ConvRecurrent

        self.conv = ConvLayer(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            activation_ff,
            norm,
            BN_momentum=BN_momentum,
        )
        self.recurrent_block = RecurrentBlock(
            input_size=out_channels, hidden_size=out_channels, kernel_size=1, activation=activation_rec
        )

    def forward(self, x, prev_state=None):

        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)




        x = self.conv(x)
        x, state = self.recurrent_block(x, prev_state)



        x = x.view(b, -1, c, h, w)


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

class STMN_Net(nn.Module):

    def __init__(self, num_classes, model_name, pretrain_choice, seq_len=8):
        super(STMN_Net, self).__init__()

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

        self.LSTM_layer1 = RecurrentConvLayer(self.plances, self.plances)
        self.LSTM_layer2 = RecurrentConvLayer(self.plances, self.plances)
        self.LSTM_layer3 = RecurrentConvLayer(self.plances, self.plances)









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