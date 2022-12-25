import os 
from os.path import join, exists
import torch 
import torch.nn as nn 
import numpy as np 
from PIL import Image

import pytorch_lightning as pl 
from torch.optim.lr_scheduler import LambdaLR 
from einops import rearrange, repeat 
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision import transforms 
from torchvision.utils import make_grid
from pytorch_lightning.utilities.distributed import rank_zero_only

from ldm.util import log_txt_as_img, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler

from ldm.models.diffusion.ddpm import DDPM




# def process(image, size, transform):
#     if size is not None:
#         image = image.resize((size, size))
    
#     if transform is not None:
#         image = transform(image)

#     return image 

class FontGenModel(DDPM):
    def __init__(self, 
                conditioning_key=None, 
                context_keys = [],
                *args, 
                **kwargs):
        self.data_root = kwargs.pop('data_root', None)
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        ckpt_path = kwargs.pop('ckpt_path', None)
        ignore_keys = kwargs.pop('ignore_keys', [])
        self.context_keys = context_keys 
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True 

        normalize = transforms.Normalize(
            mean = [0.5, 0.5, 0.5],
            std = [0.5, 0.5, 0.5]
        )
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            normalize
        ])
        
        self.style_dim = kwargs['unet_config'].params.context_dim
        self.style_embed = nn.Linear(192, self.style_dim)


    def get_gt(self, batch_content, batch_style):
        batch_im = []
        batch_size = batch_content.size(0)
        assert batch_style.size(0) == batch_size
        mask = torch.ones(batch_size, device=self.device).bool()
        for idx in range(batch_size):
            content = batch_content[idx]
            style = batch_style[idx]
            fpth = join(self.data_root, f'id_{style:d}', f'{content:04d}.png')
            if not exists(fpth):
                mask[idx] = 0
                continue 

            im = Image.open(fpth).convert('RGB')
            im = self.transform(im)
            batch_im.append(im) 
        batch_im = torch.stack(batch_im, dim=0).to(self.device)
        return batch_im, mask

    def im2seq(self, im, stride=8):
        seq = rearrange(im, 
            'b c (s1 h) (s2 w) -> b (h w) (s1 s2 c)', 
            s1=stride, s2=stride
        )
        seq = self.style_embed(seq)
        return seq
    
    def shared_step(self, batch):
        x_src = batch[self.first_stage_key].float()
        x_src = x_src.to(self.device)
        # style, context
        context = [batch[k].long().to(self.device) for k in self.context_keys]
        content, style = context 

        batch_size = x_src.size(0)
        rand_idx = torch.randperm(batch_size).to(self.device)
        style_ref = style[rand_idx]
        x, mask = self.get_gt(content, style_ref)

        x_ref = x_src[rand_idx, ...]
        x_ref = x_ref[mask, ...]
        x_src = x_src[mask, ...]
        seq_ref = self.im2seq(x_ref, stride=8)
        
        # print(x.size(), x.dtype, x_src.size(), x_ref.size())
        loss, loss_dict = self(x, c_concat= [x_src], c_crossattn = [seq_ref])
        return loss, loss_dict

    
    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=8, sample=True, return_keys=None, split='train', *args, **kwargs):
        log = dict()
        x_src = batch[self.first_stage_key].float()
        N = min(x_src.size(0), N)

        x_src = x_src[:N].to(self.device)
        context = [batch[k][:N].long().to(self.device) for k in self.context_keys]
        content, style = context 
        
        rand_idx = torch.randperm(N).to(self.device)
        style_ref = style[rand_idx]
        x, mask = self.get_gt(content, style_ref)

        x_ref = x_src[rand_idx, ...]
        x_ref = x_ref[mask, ...]
        x_src = x_src[mask, ...]
        seq_ref = self.im2seq(x_ref, stride=8)

        N = x.size(0)

        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)


        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                kwargs['c_concat'] = [ x_src ]
                kwargs['c_crossattn'] = [ seq_ref ]
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True, *args, **kwargs)
            
            log['content'] = kwargs['c_concat'][0]
            log['style'] = x_ref

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log
