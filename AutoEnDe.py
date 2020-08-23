
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 13:16:32 2020

@author: typang
"""
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def restart_init(m):
    if isinstance(m, BayesianConv):
       nn.init.kaiming_normal_(m.weight_mu.data, a=0, mode='fan_in')
       m.weight_rho.data = m.weight_rho.data.uniform_(-5,-4)
       m.bias_mu.data = m.bias_mu.data.zero_()
       m.bias_rho.data = m.bias_rho.data.uniform_(-5,-4)

class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0,1)
    
    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))
    
    def sample(self):
        epsilon = self.normal.sample(self.rho.size()).cuda()
        return self.mu + self.sigma * epsilon

class BayesianConv(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size):
        super().__init__()
        k = 1/math.sqrt(in_channels*kernel_size**2)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Weight parameters

        self.weight_mu = nn.Parameter(torch.Tensor(out_channels, in_channels,kernel_size,kernel_size).normal_(mean=0, std=k).cuda())
        self.weight_rho = nn.Parameter(torch.Tensor(out_channels, in_channels,kernel_size,kernel_size).uniform_(-5,-4).cuda())

        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        # Bias parameters

        self.bias_mu = nn.Parameter(torch.zeros(out_channels).cuda())
        self.bias_rho = nn.Parameter(torch.Tensor(out_channels).uniform_(-5,-4).cuda())
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
        # log sigma
        

    def forward(self, input):
        if self.training:
            weight = self.weight.sample()

            bias = self.bias.sample()
        else:

            weight = self.weight.mu
            bias = self.bias.mu
            
        return F.conv2d(input, weight, bias, stride = 1, padding=(1, 1))
    def log_sigma(self):
        return torch.sum(torch.log(self.weight.sigma))+torch.sum(torch.log(self.bias.sigma))
    
    def para_square(self):
        return torch.sum(self.weight.mu**2)+torch.sum(self.weight.sigma**2)+torch.sum(self.bias.mu**2)+torch.sum(self.bias.sigma**2)
class conv_layer(nn.Module):
    def __init__(self, in_channels, out_channels, is_Bayesian = True, is_sigmoid=False):
        super(conv_layer, self).__init__()
        self.is_simoid = is_sigmoid
        if is_Bayesian:
            self.conv = BayesianConv(in_channels,out_channels,kernel_size=3)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                                  padding=(1, 1))
        self.lrelu = nn.LeakyReLU()
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        if self.is_simoid:
            x = self.sigmoid(x)
        else:
            x = self.lrelu(x)
        return x
class Interpolate(nn.Module):
    def __init__(self, mode, scale_factor):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate     
        self.mode = mode
        self.scale_factor = scale_factor
    def forward(self, x):
        x = self.interp(x, mode=self.mode, scale_factor = self.scale_factor)
        return x    

class Decoder(nn.Module):
    def __init__(self, in_channels = 48, middle_channels = 96, out_channels = 1, is_Bayesian=True,img_size = []):
        super(Decoder, self).__init__()

        self.conv1 = conv_layer(in_channels, middle_channels, is_Bayesian= is_Bayesian)
        self.conv2 = conv_layer(middle_channels, middle_channels, is_Bayesian= is_Bayesian)
        self.conv3 = conv_layer(middle_channels,middle_channels,is_Bayesian= is_Bayesian)
        self.conv4 = conv_layer(middle_channels, middle_channels, is_Bayesian= is_Bayesian)
        self.conv5 = conv_layer(middle_channels, middle_channels, is_Bayesian= is_Bayesian)
        self.conv6 = conv_layer(middle_channels, middle_channels,is_Bayesian= is_Bayesian)
        self.conv7 = conv_layer(middle_channels, middle_channels, is_Bayesian= is_Bayesian)
        self.conv8 = conv_layer(middle_channels, middle_channels, is_Bayesian= is_Bayesian)
        self.conv9 = conv_layer(middle_channels, 64, is_Bayesian= is_Bayesian)
        self.conv10 = conv_layer(64, 32,is_Bayesian= is_Bayesian)
        self.conv11 = conv_layer(32, out_channels, is_sigmoid=True,is_Bayesian= is_Bayesian)
        
        padding = [(0,0,0,0) for i in range(5)]
        if img_size != []:
            padding = []
            w, h = img_size
            for i in range(5):
                w_padding = w%2
                h_padding = h%2
                w = int(w/2)
                h = int(h/2)
                padding =padding + [(0,h_padding,0,w_padding)]
                
        self.padding = padding
            
        
        
    def forward(self, x, is_dropiout=True):
            
# -----------------------------------------------
        padding = self.padding
        x = nn.functional.interpolate(x, mode='bilinear', scale_factor=2)
        x = F.pad(x,padding[4],mode='constant', value=0)
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = nn.functional.interpolate(x, mode='bilinear', scale_factor=2)
        x = F.pad(x,padding[3],mode='constant', value=0)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = nn.functional.interpolate(x, mode='bilinear', scale_factor=2)
        x = F.pad(x,padding[2],mode='constant', value=0)
        x = self.conv5(x)
        x = self.conv6(x)
        
        x = nn.functional.interpolate(x, mode='bilinear', scale_factor=2)
        x = F.pad(x,padding[1],mode='constant', value=0)
        x = self.conv7(x)
        x = self.conv8(x)

        x = nn.functional.interpolate(x, mode='bilinear', scale_factor=2)
        x = F.pad(x,padding[0],mode='constant', value=0)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        return x
    def para_square(self):
        sum = None
        for module in self.modules():
            if isinstance(module, BayesianConv):
                if sum is None:
                    sum = module.para_square()
                else:
                    sum += module.para_square()
        return sum
    def log_sigma_sum(self):
        sum = None
        for module in self.modules():
            if isinstance(module, BayesianConv):
                if sum is None:
                    sum = module.log_sigma()
                else:
                    sum += module.log_sigma()
        return sum
        
