from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim = 32,
    dim_mults = (1, 2, 4, 8),
    channels=1
)

diffusion = GaussianDiffusion(
    model,
    image_size = 64,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    loss_type = 'l1'            # L1 or L2
)

trainer = Trainer(
    diffusion,
    './pic64_all',
    train_batch_size = 32,
    train_lr = 4e-5,
    train_num_steps = 400000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = False              # whether to calculate fid during training
)

trainer.load('100')

sampled_images = diffusion.sampleFI('infer/1.png',batch_size=1)
#sampled_images = diffusion.sample(batch_size=1)
import cv2
import numpy as np
for i in range(0,len(sampled_images)):
    img=np.array(sampled_images[i][0].cpu())
    img_=(img - img.min())/(img.max()-img.min())
    cv2.imwrite('sample_'+str(i)+'.png',(img_*128).astype('uint8'))

