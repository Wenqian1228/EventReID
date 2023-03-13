import slayerSNN as snn
# from utils.utils import getNeuronConfig
import numpy as np
import torch.nn.functional as F
import torch
from torch import nn as nn
# from unet_modules import create_encoders, DoubleConv, Decoder, Abstract3DUNet, Mask3DUnet
import transforms
#################################################################################################

from PIL import Image
import matplotlib.pyplot as plt
# loader使用torchvision中自带的transforms函数
loader = transforms.Compose([
    transforms.ToTensor()])  
 
unloader = transforms.ToPILImage()



# 3 Tensor转化为PIL图片
# 输入tensor变量
# 输出PIL格式图片
def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image

#################################################################################################
def getNeuronConfig(type: str='SRMALPHA',
                    theta: float=10.,
                    tauSr: float=1.,
                    tauRef: float=1.,
                    scaleRef: float=2.,
                    tauRho: float=0.3,  # Was set to 0.2 previously (e.g. for fullRes run)
                    scaleRho: float=1.):
    """
    :param type:     neuron type
    :param theta:    neuron threshold
    :param tauSr:    neuron time constant
    :param tauRef:   neuron refractory time constant
    :param scaleRef: neuron refractory response scaling (relative to theta)
    :param tauRho:   spike function derivative time constant (relative to theta)
    :param scaleRho: spike function derivative scale factor
    :return: dictionary
    """
    return {
        'type': type,
        'theta': theta,
        'tauSr': tauSr,
        'tauRef': tauRef,
        'scaleRef': scaleRef,
        'tauRho': tauRho,
        'scaleRho': scaleRho,
    }

class NetworkBasic(torch.nn.Module):
    def __init__(self, 
                 num_classes, 
                 model_name, 
                 pretrain_choice,
                 seq_len=8,
                 netParams = snn.params('/ghome/caocz/code/Event_Camera/Event_Re_ID/VideoReID_PSTA/models/network.yaml'),
                 theta=[30, 50, 100],
                 tauSr=[1, 2, 4],
                 tauRef=[1, 2, 4],
                 scaleRef=[1, 1, 1],
                 tauRho=[1, 1, 10],
                 scaleRho=[10, 10, 100]):
        super(NetworkBasic, self).__init__()

        self.neuron_config = []
        self.neuron_config.append(
            getNeuronConfig(theta=theta[0], tauSr=tauSr[0], tauRef=tauRef[0], scaleRef=scaleRef[0], tauRho=tauRho[0],
                            scaleRho=scaleRho[0]))
        self.neuron_config.append(
            getNeuronConfig(theta=theta[1], tauSr=tauSr[1], tauRef=tauRef[1], scaleRef=scaleRef[1], tauRho=tauRho[1],
                            scaleRho=scaleRho[1]))
        self.neuron_config.append(
            getNeuronConfig(theta=theta[2], tauSr=tauSr[2], tauRef=tauRef[2], scaleRef=scaleRef[2], tauRho=tauRho[2],
                            scaleRho=scaleRho[2]))

        self.slayer1 = snn.layer(self.neuron_config[0], netParams['simulation'])
        self.slayer2 = snn.layer(self.neuron_config[1], netParams['simulation'])
        self.slayer3 = snn.layer(self.neuron_config[2], netParams['simulation'])

        # self.conv1 = self.slayer1.conv(8, 8, 5, padding=2)
        # self.conv2 = self.slayer2.conv(8, 8, 3, padding=1)

        self.conv1 = self.slayer1.conv(1, 1, 3, padding=1)
        self.conv2 = self.slayer2.conv(1, 1, 3, padding=1)

    def forward(self, spikeInput):
        psp1 = self.slayer1.psp(spikeInput)
        print('psp1=',psp1[0][0][0].shape)
        _spikes_layer_1 = psp1[0][0][0].cpu().data.numpy()
        print('_spikes_layer_1=',_spikes_layer_1.shape)
        a = np.clip(_spikes_layer_1, 0, 1) # 将numpy数组约束在[0, 1]范围内
        _spikes_layer_1 = (a * 255).astype(np.uint8) # 转换成uint8类型
        img = Image.fromarray(_spikes_layer_1)
        img.save('spikes.jpg')
        spikes_layer_1 = self.slayer1.spike(self.conv1(psp1))
        #############################################################################
        _spikes_layer_1 = spikes_layer_1[0][0][0].cpu().data.numpy()
        print('_spikes_layer_1=',_spikes_layer_1.shape)
        a = np.clip(_spikes_layer_1, 0, 1) # 将numpy数组约束在[0, 1]范围内
        _spikes_layer_1 = (a * 255).astype(np.uint8) # 转换成uint8类型
        img = Image.fromarray(_spikes_layer_1)
        img.save('spikes1.jpg')

        spikes_layer_2 = self.slayer2.spike(self.conv2(self.slayer2.psp(spikes_layer_1)))
        ##############################################################################
        _spikes_layer_2 = spikes_layer_2[0][0][0].cpu().data.numpy()
        print('_spikes_layer_2=',_spikes_layer_2.shape)
        a = np.clip(_spikes_layer_2, 0, 1) # 将numpy数组约束在[0, 1]范围内
        _spikes_layer_2 = (a * 255).astype(np.uint8) # 转换成uint8类型
        img = Image.fromarray(_spikes_layer_2)
        img.save('spikes2.jpg')
        # max_psp = torch.max(_spikes_layer_2)
        # min_psp = torch.min(_spikes_layer_2)
        # print('max_psp=',max_psp)
        # print('min_psp=',min_psp)
        # psp1_img = tensor_to_PIL(_spikes_layer_2[0][0][0])
        # psp1_img.save('psp1_img.jpg')

        b, t, c, w, h = spikes_layer_2.size()    # [16,8,3,256,128] [b, t, c, w, h]
        spikes_layer_2 = spikes_layer_2.contiguous().view(b * t, c, w, h)  # x=[128,3,256,128]
        # print('spikes_layer_2=',spikes_layer_2.shape)
        return spikes_layer_2

'''
device = torch.device('cuda:0')

netParams = snn.params('network.yaml')
model = NetworkBasic(netParams=netParams)
model = model.cuda()

# _input = torch.zeros((16,8,3,128,64))
_input = torch.rand(16,1,3,128,64)
print('_input=',_input.shape)
_input = _input.cuda()

_output = model(_input)
print('_output=',_output.shape)
'''