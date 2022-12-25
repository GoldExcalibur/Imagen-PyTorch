# from sched import scheduler
# from turtle import forward
import torch 
import torch.nn as nn 
import numpy as np 
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
from pytorch_lightning.utilities.distributed import rank_zero_only

from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler

from ldm.models.diffusion.ddpm import DDPM

class SuperResolutionModel(DDPM):
    def __init__(self, 
                conditioning_key = None,
                context_keys = [],
                *args,
                **kwargs):
        super().__init__(conditioning_key=conditioning_key,*args, **kwargs)
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        self.context_keys = context_keys
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True
    

    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        context = [self.get_input(batch, key) for key in self.context_keys]
        loss, loss_dict = self(x, c_concat=context)
        return loss, loss_dict
    
    @torch.no_grad()
    def log_images(self, batch, N=16, n_row=2, sample=True, return_keys=None, split='train', *args, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        #N = x.shape[0]
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
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
                kwargs['c_concat'] = [self.get_input(batch, key)[:N].to(self.device) for key in self.context_keys]
                
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True, *args, **kwargs)
            
            log['context'] = kwargs['c_concat'][0]
            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log


        