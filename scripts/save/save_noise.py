# save mrc projections with noise to png
import torch
import torchvision
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

import mrcfile
import scipy
import numpy as np

import linecache
import tqdm

import cv2

test_path = "../../test"


def awgnr(s:np.array, snr:int):
    """
    SNR = 10 * log(Ps/Pn)
    return Additive white Gaussian noise in Real space
    """
    # power of signal
    nzs = np.count_nonzero(s)
    nzs = s.size

    print(nzs)
    Ps = np.sum(np.power(s,2))/nzs
    print(Ps)
    Pn = Ps * (np.power(2, snr))
    print(Pn)
    noise = np.random.randn(s.size).reshape(s.shape) * np.sqrt(Pn)
    sn = s + noise
    return sn

def awgnc(s:np.array, snr:int):
    """
    SNR = 10 * log(Ps/Pn)
    return Additive white Gaussian noise in Complex space
    """
    s_r = np.real(s)
    Psr = np.sum(np.power(10, snr/ 10))
    Pnr = Psr / (np.power(10, snr/ 10))
    noise_r = np.random.randn(s.size).reshape(s.shape)*np.sqrt(Pnr)
    
    s_im = np.imag(s)
    Psim = np.sum(np.abs(s_im)**2)/ s.size
    Pnim = Psim / (np.power(10, snr /10))
    noise_im = np.random.randn(s_im.shape) * np.sqrt(Pnim)

    noise = noise_r + 1j *noise_im

    sn = s + noise

    return sn

pdblist=linecache.getlines(test_path+'/cas9/list.txt')[0].split('\n')[0].split(',')[0:1]

img_list=[]
for pdb in tqdm.tqdm(pdblist):
    f=mrcfile.open(test_path+'/cas9_volume/'+pdb+'.mrc')
    for img in f.data[0:2]:
        img=scipy.ndimage.zoom(img,64/320)
        #img_n = awgnr(img,1)
        img_=(img-np.min(img))/(np.max(img)-np.min(img))
        img_n = awgnr(img_,4)
        img_list.append(img_n.reshape(1,64,64))
    f.close()
np_list=np.array(img_list)
t_list=torch.tensor(np_list)
#for i in tqdm.tqdm(range(0,len(img_list))):
for i in tqdm.tqdm(range(0,2)):
    #torchvision.utils.save_image(t_list[i],str(i)+".pgm")
    cv2.imwrite(test_path+"/pic_noise/"+str(i)+".png",(128 - np_list[i][0]*128).astype('uint8'))
    #cv2.imwrite(test_path+"/pic_noise/"+str(i)+".tiff",- np_list[i][0])
