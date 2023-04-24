# save mrc projections to png
import torch
import torchvision
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

import mrcfile
import scipy
import numpy as np

import linecache
import tqdm

import cv2

pdblist=linecache.getlines('../cas9/list.txt')[0].split('\n')[0].split(',')

img_list=[]
for pdb in tqdm.tqdm(pdblist):
    f=mrcfile.open('../cas9_volume/'+pdb+'.mrc')
    for img in f.data:
        img=scipy.ndimage.zoom(img,64/320)
        img_=(img-np.min(img))/(np.max(img)-np.min(img))
        img_list.append(img_.reshape(1,64,64))
    f.close()
np_list=np.array(img_list)
t_list=torch.tensor(np_list)
for i in tqdm.tqdm(range(0,len(img_list))):
    #torchvision.utils.save_image(t_list[i],str(i)+".pgm")
    cv2.imwrite("./pic64_all/"+str(i)+".png",(np_list[i][0]*128).astype('uint8'))
