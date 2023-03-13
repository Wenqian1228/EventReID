import torch
import torch.nn as nn
import torch.nn.functional as F


class APM(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim=3, temperature=4, contrastive_att=True):
        super(APM, self).__init__()

        self.time_dim = time_dim 
        self.temperature = temperature
        self.contrastive_att = contrastive_att

        padding = (0, 0, 0, 0, (time_dim-1)//2, (time_dim-1)//2)
        self.padding = nn.ConstantPad3d(padding, value=0)

        self.semantic_mapping = nn.Conv3d(in_channels, out_channels, \
                                          kernel_size=1, bias=False)   

        # self.semantic_mapping = nn.Conv3d(in_channels, out_channels, \
        #                                   kernel_size=1, bias=False)          
        if self.contrastive_att:  
            self.x_mapping = nn.Conv3d(in_channels, out_channels, \
                                       kernel_size=1, bias=False)
            self.n_mapping = nn.Conv3d(in_channels, out_channels, \
                                       kernel_size=1, bias=False)
            self.contrastive_att_net = nn.Sequential(nn.Conv3d(out_channels, 1, \
                                kernel_size=1, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, t, h, w = x.size()
        N = self.time_dim

        neighbor_time_index = torch.cat([(torch.arange(0,t)+i).unsqueeze(0) for i in range(N) if i!=N//2], dim=0).t().flatten().long()

        # feature map registration
        semantic = self.semantic_mapping(x) # (b, c/16, t, h, w)
        print('semantic=',semantic.shape)
        # x_norm = F.normalize(semantic, p=2, dim=1) # (b, c/16, t, h, w)
        # x_norm_padding = self.padding(x_norm) # (b, c/16, t+2, h, w)
        # x_norm_expand = x_norm.unsqueeze(3).expand(-1, -1, -1, N-1, -1, -1).permute(0, 2, 3, 4, 5, 1).contiguous().view(-1, h*w, c//16) # (b*t*2, h*w, c/16) 
        # neighbor_norm = x_norm_padding[:, :, neighbor_time_index, :, :].permute(0, 2, 1, 3, 4).contiguous().view(-1, c//16, h*w) # (b*t*2, c/16, h*w) 

        # similarity = torch.matmul(x_norm_expand, neighbor_norm) * self.temperature # (b*t*2, h*w, h*w)
        # similarity = F.softmax(similarity, dim=-1) # (b*t*2, h*w, h*w)

        # x_padding = self.padding(x)
        # neighbor = x_padding[:, :, neighbor_time_index, :, :].permute(0, 2, 3, 4, 1).contiguous().view(-1, h*w, c)
        # neighbor_new = torch.matmul(similarity, neighbor).view(b, t*(N-1), h, w, c).permute(0, 4, 1, 2, 3) # (b, c, t*2, h, w)

        # # contrastive attention
        # if self.contrastive_att:
        #     x_att = self.x_mapping(x.unsqueeze(3).expand(-1, -1, -1, N-1, -1, -1).contiguous().view(b, c, (N-1)*t, h, w).detach())
        #     n_att = self.n_mapping(neighbor_new.detach())
        #     contrastive_att = self.contrastive_att_net(x_att * n_att)    
        #     neighbor_new = neighbor_new * contrastive_att

        # # integrating feature maps
        # x_offset = torch.zeros([b, c, N*t, h, w], dtype=x.data.dtype, device=x.device.type)
        # x_index = torch.tensor([i for i in range(t*N) if i%N==N//2])
        # neighbor_index = torch.tensor([i for i in range(t*N) if i%N!=N//2])
        # x_offset[:, :, x_index, :, :] += x
        # x_offset[:, :, neighbor_index, :, :] += neighbor_new

        # return x_offset

        return semantic

my_model = APM(8,8)
_input = torch.ones(16,8,3,256,128)
_output = my_model(_input)
print('_output=',_output.shape)