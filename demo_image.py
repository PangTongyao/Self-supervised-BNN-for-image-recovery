#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 15:05:29 2020

@author: typang
"""

import argparse
import scipy.io as sio
import numpy as np
import glob
import cv2
import torch
import torch.nn as nn
from torch.optim import Adam
from AutoEnDe import  Decoder
import os, sys
from datetime import datetime
from skimage.measure.simple_metrics import compare_psnr
from skimage.measure import compare_ssim
from util import get_cs_mearsurement
#simport torch.nn.functional as F

parser = argparse.ArgumentParser(description="CS")
parser.add_argument("--gpu", type=int, default=0, help="gpu id")
parser.add_argument("--CS_ratio", type=int, default=40, help="CS_ratio")
parser.add_argument("--iters", type=int, default=100000, help="Number of training iterations")
parser.add_argument("--in_channels", type=int, default=128, help="number of input channels")
parser.add_argument("--middle_channels", type=int, default=128, help="number of middle channels")
opts = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(opts.gpu)
torch.set_num_threads(4)

class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.__stdout__
        self.log = open(fileN, "a+")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()
#        self.close()
    def flush(self):
        self.log.flush()

def train(SIGMA,Measure,Phi,pad,block_size,image_name,MODEL_PATH,Img,gamma,is_Bayesian =True,is_MAP = False, Epsilon=1e-3):
    os.makedirs(MODEL_PATH, exist_ok=True)
    measure_num = Measure.numel()
    b,c,w,h = Img.size()
 
    net = Decoder(in_channels = opts.in_channels,middle_channels=opts.middle_channels,out_channels=c,is_Bayesian = is_Bayesian, img_size=[w,h])
    Input = torch.randn(b,opts.in_channels,int(w/32),int(h/32)).cuda()

    net.cuda()
    
    now = datetime.now()

    optimizer = Adam(net.parameters(), lr=1e-4)

    criterion = nn.MSELoss(reduction='sum')
    criterion.cuda()

    for i in range(opts.iters):


        net.train()

        net.zero_grad()

        Img_rec = net(Input)
        net_output = get_cs_mearsurement(Img_rec,Phi_tensor,pad,block_size)
        residual = criterion(net_output, Measure)
        if is_Bayesian:
            log_sigma =net.log_sigma_sum()
            para_square = net.para_square()
            loss = residual + gamma[0]*((SIGMA+Epsilon)**2)*para_square - gamma[1]*((SIGMA+Epsilon)**2)*log_sigma

        else:
            
            if is_MAP:
                weight_sum = 0
                for para in net.parameters():
                    weight_sum += gamma[0]*((SIGMA+Epsilon)**2)*torch.sum(abs(para)**2)
                loss =residual + weight_sum
            else:
                loss = residual


        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        
        if (i + 1) % 1000 == 0:
            Img_rec_single = net(Input)
            now = datetime.now()
            sys.stdout = Logger(MODEL_PATH+'results.txt')
            print(image_name, "loss in ", i + 1, ":", loss.item(),"residual:", residual.item(), now.strftime("%H:%M:%S"))

        if residual < measure_num* (SIGMA + Epsilon) ** 2:
            break

    Img_rec_aver = np.zeros(Img.size(), dtype=np.float32)
    aver_num = 100
    with torch.no_grad():
        for j in range(aver_num):
            del Img_rec

            Img_rec = net(Input)

            
            optimizer.zero_grad()
            Img_rec_cpu = Img_rec.cpu().detach().numpy()
            
            Img_rec_aver += Img_rec_cpu
                               
    Img_rec_aver = Img_rec_aver/aver_num 
    Img_rec_aver = np.squeeze(Img_rec_aver)
    
    
    Img_np = Img.cpu().numpy()
    Img_np = np.squeeze(Img_np)
    psnr_mc= compare_psnr(Img_np,Img_rec_aver,1.)
    
    Img_rec_aver = np.int32(Img_rec_aver*255)
    
    if Img_rec_aver.ndim == 3:
        
        Img_rec_aver =  Img_rec_aver.transpose(1,2,0)
        
    now = datetime.now()
    sys.stdout = Logger(MODEL_PATH+'results.txt')
    print("gamma: ",gamma,image_name,"psnr: ", psnr_mc, now.strftime("%H:%M:%S"))


    cv2.imwrite(MODEL_PATH+image_name + '_rec.png', Img_rec_aver)

    torch.save({
        'net': net,
        'net_input': Input,
        'iters': i + 1,
        'net_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, MODEL_PATH +image_name+ '.pth')

    return Img_rec_aver
       

if __name__ == '__main__':

    block_size = 33
    CS_ratio = opts.CS_ratio # 4,10,25,40
    gamma =[0.05,0.25]
    Epsilon = 1e-3
    is_Bayesian = True
    is_MAP = False
    
    Phi_data_Name = './phi/phi_0_%d_1089.mat' % CS_ratio
    Phi_data = sio.loadmat(Phi_data_Name)
    Phi_np = np.float32(Phi_data['phi'].transpose())
    Phi_tensor = torch.FloatTensor(Phi_np).cuda()


    for SIGMA in [0,10]:
        MODEL_PATH = './ImageResults/CS_ratio_%d/Sigma_%d/' % (CS_ratio,SIGMA)
        SIGMA = SIGMA/255.
        
      
        file_name = glob.glob('boats.tif')

        psnr = np.zeros(len(file_name))
        ssim = np.zeros(len(file_name))
        i =0
        for Img_Name in file_name:

            img = np.array(cv2.imread(Img_Name, -1), dtype=np.float32)/255.
            if img.ndim == 2:
                Img = np.expand_dims(img, axis=0)
            else:
                Img = img.transpose(2,0,1)
            c,w,h = Img.shape
            pad_right = block_size - w%block_size
            pad_bottom = block_size - h%block_size
            pad = (0,pad_bottom,0,pad_right)
            
            Img = np.expand_dims(Img,axis=0)
            Img_tensor =  torch.FloatTensor(Img).cuda()
            
 

            Measure = get_cs_mearsurement(Img_tensor,Phi_tensor,pad,block_size)

            Measure += torch.FloatTensor(Measure.size()).normal_(mean=0, std=SIGMA).cuda()


            img_prex = Img_Name[Img_Name.rfind("/")+1:Img_Name.rfind(".")]
            
    
           
            Img_rec = train(SIGMA,Measure,Phi_tensor,pad,block_size,img_prex,MODEL_PATH,Img_tensor,gamma,is_Bayesian,is_MAP,Epsilon)
            
            psnr[i] = compare_psnr(Img_rec/255.,img,1.)
            ssim[i] = compare_ssim(Img_rec/255.,img,data_range = 1.)
            
            sys.stdout = Logger(MODEL_PATH+'psnr.txt')
            print("gamma:",gamma,'epsilon:',Epsilon,"cs_ratio:", CS_ratio, "simga:",SIGMA, img_prex, "psnr/ssim:",psnr[i],"/",ssim[i])
            i = i+1
            
        psnr_aver = np.mean(psnr)
        ssim_aver = np.mean(ssim)
        sys.stdout = Logger(MODEL_PATH+'psnr.txt')
        print("gamma:",gamma,'epsilon:',Epsilon,"cs_ratio:", CS_ratio, "simga:",SIGMA, "average psnr/ssim:",psnr_aver,"/",ssim_aver,'\n')
           
