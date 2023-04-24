# save experiment mrc to png
import torch
import torchvision
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

import mrcfile
import scipy
import numpy as np

import linecache
import tqdm

import cv2
def run():
    img_list=[]
    f=mrcfile.open("/data/parsed2/J173/restack/batch_0_restacked.mrc")
    batchSize=100
    batchCount=0
    for img in f.data:
        if (batchCount < batchSize):
            img=scipy.ndimage.zoom(img,64/320)
            img_=(img-np.min(img))/(np.max(img)-np.min(img))
            img_list.append(img_.reshape(1,64,64))
        batchCount=batchCount+1
    f.close()
    
    np_list=np.array(img_list)
    t_list=torch.tensor(np_list)
    for i in tqdm.tqdm(range(0,len(img_list))):
        #torchvision.utils.save_image(t_list[i],str(i)+".pgm")
        cv2.imwrite("./infer/"+str(i)+".png",(np_list[i][0]*128).astype('uint8'))
