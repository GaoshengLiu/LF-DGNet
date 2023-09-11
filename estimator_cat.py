import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import scipy.io as scio
from math import sqrt
from utils import warping

class Estimator(nn.Module):
    def __init__(self, angRes):
        super(Estimator, self).__init__()
        self.angRes = angRes
        an2 = angRes*angRes
        #self.disp_estimator = disp_estimator(channel, angRes)
        
        self.disp_estimator = nn.Sequential(
            nn.Conv2d(an2,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=5,stride=1,padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=5,stride=1,padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            #nn.Conv2d(64,an2,kernel_size=1,stride=1,padding=0),
            )
        self.dis = nn.Conv2d(64,an2,kernel_size=1,stride=1,padding=0)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        an2 = self.angRes*self.angRes
        x_sv = LFsplit(x, self.angRes)        
        b, n, c, h, w = x_sv.shape
               
        estimatein = x_sv.view(b, n, h, w)
        middle = self.disp_estimator(estimatein)
        disp_target = self.dis(middle)  #  N,an2,h,w
        ind_source = ind_cal(self.angRes, self.angRes)
        '''
        warp_list = []
        warped_img = torch.zeros(b, n, h, w).type_as(estimatein)
        for k_t in range(0, an2):  # for each target view
            ind_t = torch.arange(an2)[k_t]
            for k_s in range(0, an2):
                ind_s = ind_source[k_s]
                disp = disp_target
                warped_img[:, k_s] = warping(disp[:,k_t], ind_s, ind_t, estimatein[:, k_s], self.angRes)
            warp_list.append(warped_img)
        #print(len(warp_list))
        #warp_list = torch.stack(warp_list, 1)
        #print(inter_lf.shape)        
        #warp_ave = torch.mean(warp_list, dim=1)
        #print(warp_ave.shape)
        warped_images = []
        for i in range(len(warp_list)):
            #print(warp_list[i].shape)
            warped_images.append(FormOutput(warp_list[i].unsqueeze(2)))

        #warped_img = FormOutput(inter_lf.unsqueeze(2))
        '''
        
        
        return disp_target#,  warped_images#warped_img#

def ind_cal(angout, angin):
    """
    :param features: 
    :return: 
    """

    ind_all = np.arange(angout*angout).reshape(angout, angout)        
    delt = (angout-1) // (angin-1)
    ind_source = ind_all[0:angout:delt, 0:angout:delt]
    ind_source = torch.from_numpy(ind_source.reshape(-1))
    return ind_source

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
    total = sum([param.nelement() for param in net.parameters()])
    flops, params = profile(net, inputs=(input,))
    print('   Number of parameters: %.4fM' % (total / 1e6))
    print('   Number of FLOPs: %.4fG' % (flops / 1e9))
