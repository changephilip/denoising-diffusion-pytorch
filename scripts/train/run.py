import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

import mrcfile
import scipy
import numpy as np

import linecache
import tqdm

pdblist=linecache.getlines('../cas9/list.txt')[0].split('\n')[0].split(',')[0:2]

img_list=[]
for pdb in tqdm.tqdm(pdblist):
    f=mrcfile.open('../cas9_volume/'+pdb+'.mrc')
    for img in f.data:
        img=scipy.ndimage.zoom(img,128/320)
        img_=(img-np.min(img))/(np.max(img)-np.min(img))
        img_list.append(img_.reshape(1,128,128))
    f.close()

np_img_list=np.array(img_list)
img_list=[]
training_images = torch.tensor(np_img_list)
np_img_list=[]

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 1
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
)

#training_images = torch.rand(8, 3, 128, 128) # images are normalized from 0 to 1
loss = diffusion(training_images)
loss.backward()
# after a lot of training

sampled_images = diffusion.sample(batch_size = 4)
sampled_images.shape # (4, 3, 128, 128)
