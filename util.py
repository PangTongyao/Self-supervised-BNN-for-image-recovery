import numpy as np
import torch
import torch.nn.functional as F
from skimage.measure.simple_metrics import compare_psnr
import cv2

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])




def get_cs_mearsurement(Img,Phi_input,pad,block_size):
    
    
    Img_pad = F.pad(Img, pad, mode='constant', value=0)
 
    p,c,w,h = Img_pad.size()

    Img_col = torch.reshape(Img_pad,(p,c,-1,block_size,h))
    n = Img_col.size()[2]
    Img_col = Img_col.reshape((p,c,n,block_size,-1,block_size))
    Img_col = Img_col.permute(0,1,2,4,3,5)
    Img_col = Img_col.reshape(p,c,-1,block_size*block_size)

    Img_cs = torch.matmul(Img_col, Phi_input)
    return Img_cs
        

