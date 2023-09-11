import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import scipy.io as scio
from math import sqrt
from numpy import clip
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
#from dcn.modules.deform_conv import DeformConv
class Net(nn.Module):
    def __init__(self, angRes, factor):
        super(Net, self).__init__()
        n_blocks, channel = 4, 32
        self.factor = factor
        self.angRes = angRes
        self.FeaExtract = FeaExtract(channel)
        self.Upda_1 = C_DGFM_MFR(channel, angRes)
        self.Upda_2 = C_DGFM_MFR(channel, angRes)
        self.Upda_3 = C_DGFM_MFR(channel, angRes)
        self.Upda_4 = C_DGFM_MFR(channel, angRes)
        self.Reconstruct = CascadedBlocks(n_blocks, 4*channel)#Heirarchical Feature Fusion
        self.UpSample = Upsample(channel, factor)
        
    def forward(self, x, disp_target):
        x_sv = LFsplit(x, self.angRes)        
        b, n, c, h, w = x_sv.shape
        
        multi_view_fea_initial = self.FeaExtract(x_sv)         
        
        x_sv = x_sv.contiguous().view(b*n, -1, h, w)
        x_upscale = F.interpolate(x_sv, scale_factor=self.factor, mode='bicubic', align_corners=False)
        _, c, h_, w_ = x_upscale.shape
        x_upscale = x_upscale.unsqueeze(1).contiguous().view(b, -1, c, h_, w_)
               
        multi_view_fea_0 = self.Upda_1(multi_view_fea_initial, disp_target)        
        multi_view_fea_1 = self.Upda_2(multi_view_fea_0, disp_target)        
        multi_view_fea_2 = self.Upda_3(multi_view_fea_1, disp_target)        
        multi_view_fea_3 = self.Upda_4(multi_view_fea_2, disp_target)        
        multi_view_feas = torch.cat((multi_view_fea_0, multi_view_fea_1, multi_view_fea_2, multi_view_fea_3), 2)        

        fused_multi_view_fea = self.Reconstruct(multi_view_feas)
        out_sr = self.UpSample(fused_multi_view_fea)        
        out = FormOutput(out_sr) + FormOutput(x_upscale)

        return out

class Upsample(nn.Module):
    '''
    Upsampling Block
    '''
    def __init__(self, channel, factor):
        super(Upsample, self).__init__()
        self.upsp = nn.Sequential(
            nn.Conv2d(4*channel, channel * factor * factor, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(factor),
            nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        b, n, c, h, w = x.shape
        x = x.contiguous().view(b*n, -1, h, w)
        out = self.upsp(x)
        _, _, H, W = out.shape
        out = out.contiguous().view(b, n, -1, H, W)
        return out

class FeaExtract(nn.Module):
    '''
    Feature Extraction
    '''
    def __init__(self, channel):
        super(FeaExtract, self).__init__()
        self.FEconv = nn.Conv2d(1, channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.FERB_1 = RB(channel)

    def forward(self, x_sv):
        b, n, r, h, w = x_sv.shape
        x_sv = x_sv.contiguous().view(b*n, -1, h, w)
        buffer_sv_0 = self.FEconv(x_sv)
        buffer_sv = self.FERB_1(buffer_sv_0)
        _, c, h, w = buffer_sv.shape
        buffer_sv = buffer_sv.unsqueeze(1).contiguous().view(b, -1, c, h, w)#.permute(0,2,1,3,4)  # buffer_sv:  B, N, C, H, W

        return buffer_sv
class RB(nn.Module):
    def __init__(self, channel):
        super(RB, self).__init__()
        self.conv01 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.conv02 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        buffer = self.conv01(x)
        buffer = self.lrelu(buffer)
        buffer = self.conv02(buffer)
        return buffer + x
class Dgfm(nn.Module):
    '''
    Disparity-guided feature modulation
    '''
    def __init__(self, channels, angRes):
        super(Dgfm, self).__init__()
        #self.interpolate = nn.functional.interpolate
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 =  nn.Conv2d(1, channels, kernel_size=3, padding=1)   
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    def forward(self, disparity_score, x_init):
        tmp = self.relu1(self.conv1(disparity_score))
        gamma = self.conv2(tmp)
        beta = self.conv3(tmp)
        x = x_init * gamma + beta
        return x
class C_DGFM_MFR(nn.Module):
    '''
    cascaded Disparity-guided feature modulation and multi-view feature recalibration
    '''
    def __init__(self, channel, angRes, last=False):
        super(C_DGFM_MFR, self).__init__()
        self.conv_f1 = nn.Conv2d(angRes*angRes*channel, angRes*angRes*channel, kernel_size=1, stride=1, padding=0)
        self.re0 = Dgfm(channel,angRes)
        self.last = last
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.att = Tripleatt(angRes, channel)

    def init_offset(self):
        self.conv_off.weight.data.zero_()
        self.conv_off.bias.data.zero_()

    def forward(self, multi_view_fea, disparity):
        b, n, c, h, w = multi_view_fea.shape
        buffer = []
        for i in range(n):
            view_wise_fea  = multi_view_fea[:, i, :, :, :].contiguous()
            disparity_score = disparity[:, i, :, :].contiguous().unsqueeze(1)
            modu_fea = self.re0(disparity_score, view_wise_fea)
            buffer.append(modu_fea)
        buffer = torch.stack(buffer, dim=1)       # B, N*C, H, W
        Att_fea = self.att(buffer.contiguous().view(b*n, c, h, w)).view(b, n, c, h, w)
        Enhance_fea = self.lrelu(self.conv_f1(Att_fea.view(b, n*c, h, w)))#feature enhancement
        
        out = Enhance_fea.contiguous().view(b, n, -1, h, w) 

        return out
class Tripleatt(nn.Module):
    '''
    Triple attention
    '''
    def __init__(self, an, ch):
        super(Tripleatt, self).__init__()
                
        self.relu = nn.ReLU(inplace=True)
        
        self.spaconv_s = nn.Conv2d(in_channels = ch, out_channels = ch, kernel_size = (3,3), stride = 1, padding = 1, dilation=1)
        self.spaconv_c = nn.Conv2d(in_channels = ch, out_channels = ch, kernel_size = (3,3), stride = 1, padding = 1, dilation=1)
        self.angconv = nn.Conv2d(in_channels = ch, out_channels = ch, kernel_size = (3,3), stride = 1, padding = (1,1))
        self.satt = nn.Sequential(
                nn.Conv2d(in_channels = 2, out_channels = 1, kernel_size = 7, stride = 1, padding = 3),
                nn.Sigmoid()
                )
        self.aatt = nn.Sequential(
                nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 3, stride = 1, padding = 1),
                #nn.Sigmoid()
                )
        self.catt = nn.Sequential(
                nn.Conv2d(in_channels = ch, out_channels = ch//4, kernel_size = 1, stride = 1, padding = 0),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels = ch//4, out_channels = ch, kernel_size = 1, stride = 1, padding = 0),
                nn.Sigmoid()
                )
        self.an = an
        self.an2 = an*an
        self.fuse = nn.Conv2d(in_channels = ch, out_channels = ch, kernel_size = (3,3), stride = 1, padding = 1, dilation=1)
        self.softmax = nn.Softmax(dim=2)
    
    def forward(self,x):

        N,c,h,w = x.shape
        N = N // self.an2
        
        ###spatial attention
        s_out = self.spaconv_s(x)
        
        fm_pool = torch.cat([torch.mean(s_out, dim=1, keepdim=True), torch.max(s_out, dim=1, keepdim=True)[0]], dim=1)
        att = self.satt(fm_pool)
        s_out = s_out * att    
        
        
        ###angular attention       
        a_in = x.view(N,self.an2,c,h*w)
        a_in = torch.transpose(a_in,1,3)
        a_in = a_in.contiguous().view(N*h*w,c,self.an,self.an)
        a_out = self.angconv(a_in)
        fm_pool = torch.mean(a_out, dim=1, keepdim=True)        
        fm_pool = self.aatt(fm_pool)
        att = self.softmax(fm_pool.contiguous().view(N*h*w,1,self.an2))
        att = att.contiguous().view(N*h*w,1,self.an, self.an)        
        a_out = a_out * att
        a_out = a_out.view(N,h*w,c,self.an2)
        a_out = torch.transpose(a_out,1,3)
        a_out = a_out.contiguous().view(N*self.an2,c,h,w)
        
        ###channel attention
        c_out = self.spaconv_c(x)
        fm_pool = F.adaptive_avg_pool2d(c_out, (1, 1))
        att = self.catt(fm_pool)
        c_out = c_out * att
        
        out = self.fuse(s_out + a_out + c_out)

        return out
    
class Racb(nn.Module):
    '''
    Residual asymetric convolution block
    '''
    def __init__(self, channel):
        super(Racb, self).__init__()
        self.conv01 =   nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv01_1 = nn.Conv2d(channel, channel, kernel_size=(1,3), stride=1, padding=(0,1))
        self.conv01_2 = nn.Conv2d(channel, channel, kernel_size=(3,1), stride=1, padding=(1,0))
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.conv02 = nn.Conv2d(channel*3, channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        #b, n, c, h, w = x.shape
        buffer_0 = self.lrelu(self.conv01(x))
        buffer_1 = self.lrelu(self.conv01_1(x))
        buffer_2 = self.lrelu(self.conv01_2(x))
        buffer = torch.cat((buffer_0, buffer_1, buffer_2), 1)
        buffer = self.conv02(buffer)#.contiguous().view(b, n, -1, h, w)
        return buffer + x
class CascadedBlocks(nn.Module):
    '''
    Hierarchical feature fusion
    '''
    def __init__(self, n_blocks, channel):
        super(CascadedBlocks, self).__init__()
        self.n_blocks = n_blocks
        body = []
        for i in range(n_blocks):
            body.append(RDB(channel,4,channel//4))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        for i in range(self.n_blocks):
            x = self.body[i](x)
        return x

class one_conv(nn.Module):
    '''
    1x1 Dense convolution in RDB
    '''
    def __init__(self, G0, G):
        super(one_conv, self).__init__()
        self.conv = nn.Conv2d(G0, G, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        output = self.relu(self.conv(x))
        return torch.cat((x, output), dim=1)
class RDB(nn.Module):
    '''
    Residual dense block
    '''
    def __init__(self, G0, C, G):
        super(RDB, self).__init__()
        self.convin = nn.Conv2d(G0, G0, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        convs = []
        for i in range(C):
            convs.append(one_conv(G0+i*G, G))
        self.conv = nn.Sequential(*convs)
        self.LFF = nn.Conv2d(G0+C*G, G0, kernel_size=1, stride=1, padding=0, bias=True)
    def forward(self, x):
        b, n, c, h, w = x.shape
        buffer = x.contiguous().view(b*n, -1, h, w)
        buffer = self.relu(self.convin(buffer))
        out = self.conv(buffer)
        lff = self.LFF(out)
        lff = lff.contiguous().view(b, n, -1, h, w)
        return lff + x

def LFsplit(data, angRes):
    b, _, H, W = data.shape
    h = int(H/angRes)
    w = int(W/angRes)
    data_sv = []
    for u in range(angRes):
        for v in range(angRes):
            data_sv.append(data[:, :, u*h:(u+1)*h, v*w:(v+1)*w])

    data_st = torch.stack(data_sv, dim=1)
    return data_st


def FormOutput(x_sv):
    b, n, c, h, w = x_sv.shape
    angRes = int(sqrt(n+1))
    out = []
    kk = 0
    for u in range(angRes):
        buffer = []
        for v in range(angRes):
            buffer.append(x_sv[:, kk, :, :, :])
            kk = kk+1
        buffer = torch.cat(buffer, 3)
        out.append(buffer)
    out = torch.cat(out, 2)

    return out


if __name__ == "__main__":
    net = Net(5, 2).cuda()
    from thop import profile
    input = torch.randn(1, 1, 160, 160).cuda()
    dis = torch.randn(1, 25, 32, 32).cuda()
    total = sum([param.nelement() for param in net.parameters()])
    flops, params = profile(net, inputs=(input,dis))
    print('   Number of parameters: %.4fM' % (total / 1e6))
    print('   Number of FLOPs: %.4fG' % (flops / 1e9))