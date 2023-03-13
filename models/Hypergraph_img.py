import torch
from  torch import nn
import  torch.utils.model_zoo as model_zoo
from torch.nn import functional as F

from models.backbone.resnet import *
from models.STAM import STAM
import sys
import random
from .resnet import ResNet, Bottleneck
import numpy as np
import scipy.sparse as sp
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
class NearestConvolution(nn.Module):
    """
    Use both neighbors on graph structures and neighbors of nearest distance on embedding space
    """
    def __init__(self, dim_in, dim_out):
        super(NearestConvolution, self).__init__()

        self.kn = 3
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.fc = nn.Linear(self.dim_in, self.dim_out, bias=False)
        self.dropout = nn.Dropout(p=0.1)

        self.trans = ConvMapping(self.dim_in, self.kn)

    def _nearest_select(self, feats):
        b = feats.size()[0]
        N = feats.size()[1]
        dis = NearestConvolution.cos_dis(feats)
        _, idx = torch.topk(dis, self.kn, dim=2)
        #k_nearest = torch.stack([feats[idx[i]] for i in range(N)], dim=0)
        k_nearest = torch.stack([torch.stack([feats[j, idx[j, i]] for i in range(N)], dim=0) for j in range(b)], dim=0)                                        # (b, N, self.kn, d)
        return k_nearest

    @staticmethod
    def cos_dis(X):
        """
        cosine distance
        :param X: (b, N, d)
        :return: (b, N, N)
        """
        X = nn.functional.normalize(X, dim=2, p=2)
        XT = X.transpose(1, 2)                             #(b, d, N)
        return torch.bmm(X, XT)                            #(b, N, N)
        #return torch.matmul(X, XT)

    def forward(self, feats, edge_dict):
        """
        :param feats:
        :param edge_dict:
        :return:
        """
        x = feats                                           # (b, N, d)
        x1 = self._nearest_select(x)                        # (b, N, kn, d)
        x_list = []
        for i in range(x1.shape[0]):
            x = self.trans(x1[i])                                  # (N, d)
            x = F.relu(self.fc(self.dropout(x)))       # (N, d')
            x_list.append(x)
        x = torch.stack(x_list, dim=0)                      #(b, N, d')
        return x

class BatchedGraphSAGE(nn.Module):
    def __init__(self, infeat, outfeat, use_bn=True, mean=False, add_self=False):
        super(BatchedGraphSAGE, self).__init__()
        self.add_self = add_self
        self.use_bn = use_bn
        self.mean = mean
        self.W = nn.Linear(infeat, outfeat, bias=True)
        nn.init.xavier_uniform_(self.W.weight, gain=nn.init.calculate_gain('relu'))
        if self.use_bn:
            self.bn = nn.BatchNorm1d(outfeat)
            #self.bn = nn.BatchNorm1d(16)

    def forward(self, x, adj):
        #print(adj.shape)
        #print(x.shape)
        h_k_N = torch.matmul(adj, x)
        h_k = self.W(h_k_N)
        h_k = F.normalize(h_k, dim=2, p=2)
        h_k = F.relu(h_k)
        #print(h_k.shape)
        if self.use_bn:
            #self.bn = nn.BatchNorm1d(h_k.size(1))
            h_k = self.bn(h_k.permute(0,2,1).contiguous())
            #print(h_k.shape)
            h_k = h_k.permute(0, 2, 1)
            #print(h_k.shape)

        return h_k

def sampler_fn(adj):
    n = adj.size(0)
    #print(adj.data)
    adj = adj.data>0
    n_max = adj.sum(dim=0).max() - 1
    nei = []
    for i in range(n):
        tmp = [j for j in range(n) if adj[i,j]>0 and j != i]
        if len(tmp) != n_max:
            tmp += tmp
            random.shuffle(tmp)
            tmp = tmp[0:n_max]
        nei += tmp
    return nei

class BatchedGAT_cat1(nn.Module):
    def __init__(self, infeat, outfeat, use_bn=True, mean=False, add_self=False):
        super(BatchedGAT_cat1, self).__init__()
        self.add_self = add_self
        self.use_bn = use_bn
        self.mean = mean
        self.aggregator = True
        self.num_head = 1

        self.W_x = nn.Linear(infeat, outfeat, bias=True)
        nn.init.xavier_uniform_(self.W_x.weight, gain=nn.init.calculate_gain('relu'))

        self.W_neib = nn.Linear(infeat, outfeat, bias=True)
        nn.init.xavier_uniform_(self.W_neib.weight, gain=nn.init.calculate_gain('relu'))

        self.W_a = nn.ModuleList([nn.Linear(2*infeat, 1, bias=False) for i in range(self.num_head)])
        for i in range(self.num_head):
            nn.init.xavier_uniform_(self.W_a[i].weight, gain=nn.init.calculate_gain('relu'))

        self.alpha = 0.2
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        if self.use_bn:
            self.bn = nn.BatchNorm1d((self.num_head + 1) * outfeat)

    def forward(self, x, adj):
        b = x.size(0)
        h_k_list = []
        #x = self.W_x(x)

        sample_size = adj.size(0)
        assert(sample_size == x.size(1))
        idx_neib = sampler_fn(adj)
        x_neib = x[:, idx_neib, :].contiguous()
        x_neib = x_neib.view(b, sample_size, -1, x_neib.size(2))
        #print(x_neib.shape)

        a_input = torch.cat((x.unsqueeze(2).repeat(1, 1, x_neib.size(2), 1), x_neib), 3)
        #print(a_input.shape)
        h_k = self.W_x(x)
        #h_k_junk = self.W_x(x[i, sample_size:, :].unsqueeze(0))
        for j in range(self.num_head):
            e = self.leakyrelu(self.W_a[j](a_input).squeeze(3))
            #print(e.shape)
            attention = F.softmax(e, dim=2)
            #print(attention.shape)
            h_prime = torch.matmul(attention.unsqueeze(2), x_neib)
            #print(h_k.shape)
            #print(h_prime.shape)
            h_k = torch.cat((h_k, self.W_neib(h_prime.squeeze(2))), 2)
            #h_k_junk = torch.cat((h_k_junk, self.W_neib(x[i, sample_size:, :].unsqueeze(0))), 2)
        #h_k = torch.cat((h_k, h_k_junk), 1)
        h_k_list.append(h_k)
        h_k_f = torch.cat(h_k_list, dim=2)

        h_k_f = F.normalize(h_k_f, dim=2, p=2)
        h_k_f = F.relu(h_k_f)
        if self.use_bn:
            h_k_f = self.bn(h_k_f.permute(0, 2, 1).contiguous())
            h_k_f = h_k_f.permute(0, 2, 1)


        return h_k_f

class BatchedGAT_cat1Temporal(nn.Module):
    def __init__(self, infeat, outfeat, use_bn=True, mean=False, add_self=False):
        super(BatchedGAT_cat1Temporal, self).__init__()
        self.add_self = add_self
        self.use_bn = use_bn
        self.mean = mean
        self.aggregator = True
        self.num_head = 1

        self.W_x = nn.Linear(infeat, outfeat, bias=True)
        nn.init.xavier_uniform_(self.W_x.weight, gain=nn.init.calculate_gain('relu'))

        self.W_neib = nn.Linear(infeat, outfeat, bias=True)
        nn.init.xavier_uniform_(self.W_neib.weight, gain=nn.init.calculate_gain('relu'))

        self.W_a = nn.ModuleList([nn.Linear(2*infeat, 1, bias=False) for i in range(self.num_head)])
        for i in range(self.num_head):
            nn.init.xavier_uniform_(self.W_a[i].weight, gain=nn.init.calculate_gain('relu'))

        self.alpha = 0.2
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        if self.use_bn:
            self.bn = nn.BatchNorm1d((self.num_head*3 + 1) * outfeat)

    def forward(self, x, adj1, adj2, adj3):
        b = x.size(0)
        h_k_list = []
        #x = self.W_x(x)

        sample_size1 = adj1.size(0)
        assert(sample_size1 == x.size(1))
        idx_neib1 = sampler_fn(adj1)
        x_neib1 = x[:, idx_neib1, :].contiguous()
        x_neib1 = x_neib1.view(b, sample_size1, -1, x_neib1.size(2))
        a_input1 = torch.cat((x.unsqueeze(2).repeat(1, 1, x_neib1.size(2), 1), x_neib1), 3)

        sample_size2 = adj2.size(0)
        assert(sample_size2 == x.size(1))
        idx_neib2 = sampler_fn(adj2)
        x_neib2 = x[:, idx_neib2, :].contiguous()
        x_neib2 = x_neib2.view(b, sample_size2, -1, x_neib2.size(2))
        a_input2 = torch.cat((x.unsqueeze(2).repeat(1, 1, x_neib2.size(2), 1), x_neib2), 3)

        sample_size3 = adj3.size(0)
        assert(sample_size3 == x.size(1))
        idx_neib3 = sampler_fn(adj3)
        x_neib3 = x[:, idx_neib3, :].contiguous()
        x_neib3 = x_neib3.view(b, sample_size3, -1, x_neib3.size(2))
        a_input3 = torch.cat((x.unsqueeze(2).repeat(1, 1, x_neib3.size(2), 1), x_neib3), 3)

        h_k = self.W_x(x)
        for j in range(self.num_head):
            e1 = self.leakyrelu(self.W_a[j](a_input1).squeeze(3))
            attention1 = F.softmax(e1, dim=2)
            h_prime1 = torch.matmul(attention1.unsqueeze(2), x_neib1)

            e2 = self.leakyrelu(self.W_a[j](a_input2).squeeze(3))
            attention2 = F.softmax(e2, dim=2)
            h_prime2 = torch.matmul(attention2.unsqueeze(2), x_neib2)

            e3 = self.leakyrelu(self.W_a[j](a_input3).squeeze(3))
            attention3 = F.softmax(e3, dim=2)
            h_prime3 = torch.matmul(attention3.unsqueeze(2), x_neib3)

            h_k = torch.cat((h_k, self.W_neib(h_prime1.squeeze(2)), self.W_neib(h_prime2.squeeze(2)),  self.W_neib(h_prime3.squeeze(2))), 2)
        #h_k_list.append(h_k)

        #h_k_f = torch.cat(h_k_list, dim=2)
        h_k_f = h_k

        h_k_f = F.normalize(h_k_f, dim=2, p=2)
        h_k_f = F.relu(h_k_f)

        if self.use_bn:
            h_k_f = self.bn(h_k_f.permute(0, 2, 1).contiguous())
            h_k_f = h_k_f.permute(0, 2, 1)


        return h_k_f

class BatchedGraphSAGEDynamicRangeMean1(nn.Module):
    def __init__(self, infeat, outfeat, use_bn=True, mean=False, add_self=False):
        super(BatchedGraphSAGEDynamicRangeMean1, self).__init__()
        self.add_self = add_self
        self.use_bn = use_bn
        self.mean = mean
        self.aggregator = True
        #print(infeat,outfeat)
        self.W_x = nn.Linear(infeat, outfeat, bias=True)
        nn.init.xavier_uniform_(self.W_x.weight, gain=nn.init.calculate_gain('relu'))

        self.W_neib = nn.Linear(infeat, outfeat, bias=True)
        nn.init.xavier_uniform_(self.W_neib.weight, gain=nn.init.calculate_gain('relu'))

        if self.use_bn:
            self.bn = nn.BatchNorm1d(2*outfeat)
            #self.bn = nn.BatchNorm1d(16)

        self.kn = 3

    def forward(self, x, adj, p, t):
        #print(adj.shape)
        #print(x.shape)
        # x: (b, N, d)
        b = x.size()[0]
        N = x.size()[1]
        k_nearest_list = []
        tk = self.kn
        for i in range(int(N/p)):
            idx_start = max(0, i-t)
            idx_end = min(i+t+1, int(N/p))
            tmp_x = x[:,idx_start*p:idx_end*p,]
            dis = NearestConvolution.cos_dis(tmp_x)
            if i==0:
              tk = min(dis.shape[2], self.kn)
              #print(tk)
            _, idx = torch.topk(dis, tk, dim=2)
            #print(tmp_x.shape)
            #print(idx,idx.shape)
            #print(dis,dis.shape)
            k_nearest = torch.stack([torch.stack([tmp_x[j, idx[j, i]] for i in range(p*(idx_end-idx_start))], dim=0) for j in range(b)], dim=0) #(b, x*p, kn, d)
            #print(k_nearest)
            k_nearest_list.append(k_nearest[:,p*(i-idx_start):p*(i-idx_start+1),])
        k_nearest = torch.cat(k_nearest_list, dim=1) #(b,N, kn, d)
        x_neib = k_nearest[:,:,1:,].contiguous()

        #x_neib = x_neib.view(x.size(0), x.size(1), -1, x_neib.size(2))
        x_neib = x_neib.mean(dim=2)
        #print(k_nearest.shape)
        #x_cmp = x - k_nearest[:,:,0]
        #print(torch.sum(x_cmp)) 

        h_k = torch.cat((self.W_x(x), self.W_neib(x_neib)), 2)

        h_k = F.normalize(h_k, dim=2, p=2)
        h_k = F.relu(h_k)
        #print(h_k.shape)
        if self.use_bn:
            #self.bn = nn.BatchNorm1d(h_k.size(1))
            h_k = self.bn(h_k.permute(0,2,1).contiguous())
            #print(h_k.shape)
            h_k = h_k.permute(0, 2, 1)
            #print(h_k.shape)

        return h_k


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

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def build_adj_full_d(t=4, p=4, d=1):
    rows = []
    cols = []
    for dd in range(d):
        for j in range(t-dd-1):
            for i in range(p):
                rows += [i+j*p for k in range(p)]
                cols += range((j+1+dd)*p, (j+1+dd)*p+p)
    data = np.ones(len(rows))
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(t*p, t*p), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #print(adj)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = torch.FloatTensor(np.array(adj.todense()))
    return adj


##############################################################################
##############################################################################
##############################################################################
class OSNet(nn.Module):

    def __init__(self, num_classes, model_name, pretrain_choice, seq_len=8):
        super(OSNet, self).__init__()

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

        self.os_conv_layer1 = BatchedGraphSAGEDynamicRangeMean1(2048, 512)
        self.os_conv_layer2 = BatchedGraphSAGEDynamicRangeMean1(2048, 512)
        self.os_conv_layer3 = BatchedGraphSAGEDynamicRangeMean1(2048, 512)

        self.p1 = 4. #4.
        self.p2 = 8.
        #self.p2 = 6.
        self.p3 = 2.
        self.adj1_d1 = build_adj_full_d(8, int(self.p1), 1) 
        self.adj1_d2 = build_adj_full_d(8, int(self.p1), 2)
        self.adj1_d3 = build_adj_full_d(8, int(self.p1), 3)
        self.adj2_d1 = build_adj_full_d(8, int(self.p2), 1)
        self.adj2_d2 = build_adj_full_d(8, int(self.p2), 2)
        self.adj2_d3 = build_adj_full_d(8, int(self.p2), 3)
        self.adj3_d1 = build_adj_full_d(8, int(self.p3), 1)
        self.adj3_d2 = build_adj_full_d(8, int(self.p3), 2)
        self.adj3_d3 = build_adj_full_d(8, int(self.p3), 3)
        self.adj1_d1.requires_gradient = False
        self.adj2_d1.requires_gradient = False
        self.adj3_d1.requires_gradient = False
        self.adj1_d2.requires_gradient = False
        self.adj2_d2.requires_gradient = False
        self.adj3_d2.requires_gradient = False
        self.adj1_d3.requires_gradient = False
        self.adj2_d3.requires_gradient = False
        self.adj3_d3.requires_gradient = False

    def forward(self, x, pids=None, camid=None):    # x=[16,8,3,256,128]

        b, t, c, w, h = x.size()    # [16,8,3,256,128] [b, t, c, w, h]
        x = x.contiguous().view(b * t, c, w, h)  # x=[128,3,256,128]
        # 调用里面的模块，然后提取特征
        
        # feature_map1 =  self.base.conv1(x)
        # feature_map2 = self.base.conv2(feature_map1)
        
        feat_map = self.base(x)  # (b * t, c, 16, 8)  feat_map= torch.Size([128, 2048, 16, 8])      
        # print('feat_map=',feat_map.shape)

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