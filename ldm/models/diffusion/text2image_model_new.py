import os 
from os.path import exists
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

from ldm.models.diffusion.ddpm import DDPM 

# text model part 
from transformers import T5Tokenizer, T5EncoderModel, T5Config, BertTokenizer 

def load_text_model(cp_dir, eval_mode=True):
    assert exists(cp_dir), f'pretrained text dir {cp_dir:s} not exists !'
    tokenizer = BertTokenizer.from_pretrained(cp_dir)
    encoder = T5EncoderModel.from_pretrained(cp_dir)
    
    if eval_mode: 
        for p in encoder.parameters():
            p.requires_grad = False 
            # p.detach_()
        encoder.eval()
    else:
        encoder.train()
    return tokenizer, encoder 

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob


class Text2ImageModel(DDPM):
    def __init__(self, 
        conditioning_key = None, 
        context_keys = [], 
        text_model_pth = None, 
        text_pad_type = 'max_length',
        max_len = 256,
        cond_drop_prob = 0.1,
        *args, 
        **kwargs):
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        ckpt_path = kwargs.pop('ckpt_path', None)
        ignore_keys = kwargs.pop('ignore_keys', [])
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True 

        self.context_keys = context_keys 
        # load pretrained text model (t5)
        assert text_pad_type in ['max_length', 'longest', 'do_not_pad']
        self.text_pad_type = text_pad_type 
        self.max_len = max_len 
        self.tokenizer, self.text_model = load_text_model(text_model_pth, eval_mode=True)
        self.cond_drop_prob = cond_drop_prob # dropout for jointly train uncond & cond model 
        self.cond_dim = kwargs['unet_config'].params.model_channels
        self.null_text_embed = nn.Parameter(torch.randn(1, self.max_len, self.cond_dim), requires_grad=True)
        self.text_to_cond = nn.Linear(768, self.cond_dim)
    
    @torch.no_grad()
    def get_text(self, batch, k):
        batch_text = batch[k]
        encoded = self.tokenizer.batch_encode_plus(
            batch_text, return_tensors='pt', padding=self.text_pad_type, 
            max_length=self.max_len, truncation=True,
        )

        # device = next(self.text_model.parameters()).device
        input_ids = encoded.input_ids.to(self.device)
        attn_mask = encoded.attention_mask.to(self.device)

        output = self.text_model(input_ids=input_ids, attention_mask=attn_mask)
        text_embed = output.last_hidden_state #.detach()
        return text_embed #, attn_mask.bool()
    
    def get_text_token(self, text_embed, cond_prob_drop):
        text_embed = self.text_to_cond(text_embed) #b,l,c

        batch_size = text_embed.size(0)
        text_mask = prob_mask_like( (batch_size,), 1.0 - cond_prob_drop, self.device )
        text_mask = rearrange(text_mask, 'b -> b 1 1')

        # print(text_mask.size(), text_embed.size(), self.null_text_embed.size())
        text_embed = torch.where(
            text_mask, # b,1,1
            text_embed, # b,l,c
            self.null_text_embed, # 1,l,c
        )
        # print(text_embed.size(), text_embed.requires_grad, text_mask.sum() / float(batch_size))

        return text_embed #, attn_mask.bool()

    def shared_step(self, batch, **kwargs):
        x = self.get_input(batch, self.first_stage_key)
        context = [ self.get_text(batch, key) for key in self.context_keys]
        # drop_p = kwargs['cond_drop_prob'] if 'cond_drop_prob' in kwargs else self.cond_drop_prob
        # context = [ self.get_text_token(c, drop_p) for c in context]
        # print(x.requires_grad, [tmp.requires_grad for tmp in context])
        loss, loss_dict = self(x, c_crossattn = context)
        return loss, loss_dict 

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None,  split='train', *args, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
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

        log['diffusion_row'] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row 
            with self.ema_scope("Plotting"):
                kwargs['c_crossattn'] = [self.encode_text(batch, key)[:N].to(self.device) for key in self.context_keys]
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True, *args, **kwargs)
                log['samples'] = samples 
                log['denoise_row'] = self._get_rows_from_list(denoise_row)
        
        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}

        return log 
        
