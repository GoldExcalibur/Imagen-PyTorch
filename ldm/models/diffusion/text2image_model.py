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
from transformers import T5Tokenizer, T5EncoderModel, T5Config, BertTokenizer, BartModel
from utils.utils import get_fname, right_pad_dims_to

name2dim = {
    't5_base_chn': 768,
    't5_base': 768,
    't5_xl': 2048,
    'mt5_xl': 2048,
    'bart_large_chn': 1024,
}

def load_text_model(cp_dir, eval_mode=True):
    assert exists(cp_dir), f'pretrained text dir {cp_dir:s} not exists !'
    model_name = get_fname(cp_dir)
        
    if 't5' in model_name:
        try:
            tokenizer = T5Tokenizer.from_pretrained(cp_dir)
        except:
            tokenizer = BertTokenizer.from_pretrained(cp_dir)
        encoder = T5EncoderModel.from_pretrained(cp_dir)
    elif 'bart' in model_name:
        tokenizer = BertTokenizer.from_pretrained(cp_dir)
        encoder = BartModel.from_pretrained(cp_dir)
    else:
        raise NotImplementedError(f'{model_name:s} not supported !')

    if eval_mode: 
        for p in encoder.parameters():
            p.detach_()
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

def dynamic_threshold(x, percentile=0.9):
    s = torch.quantile(
            rearrange(x, 'b ... -> b (...)').abs(),
            percentile,
            dim = -1
        ) # bsize 

    s.clamp_(min = 1.)
    s = right_pad_dims_to(x, s)
    return s 

class Text2ImageModel(DDPM):
    def __init__(self, 
        conditioning_key = None, 
        context_keys = [], 
        text_model_pth = None, 
        text_pad_type = 'max_length',
        max_len = 256,
        cond_drop_prob = 0.1,
        cond_scale = 3.0,
        clip_method = 'dynamic',
        dynamic_percentile = 0.9,
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
        self.text_model_pth = text_model_pth 
        self.tokenizer, self.text_model = load_text_model(self.text_model_pth, eval_mode=True)
        # classifier free guidance (cond_drop_prob for train & cond_scale for text)
        self.cond_drop_prob = cond_drop_prob 
        assert cond_scale > 1.0, 'cond_scale {:.3f} should be larger than 1 !'.format(cond_scale)
        self.cond_scale = cond_scale  

        self.pre_dim = name2dim[ get_fname(self.text_model_pth) ]
        self.cond_dim = kwargs['unet_config'].params.context_dim #model_channels
        self.text_to_cond = nn.Linear(self.pre_dim, self.cond_dim)
        self.null_text_embed = nn.Parameter(torch.randn(1, self.max_len, self.cond_dim), requires_grad=True)
        # static or dynamic thresholding as in imagen
        assert clip_method in ['static', 'dynamic']
        self.clip_method = clip_method
        self.dynamic_percentile = dynamic_percentile
    
    @torch.no_grad()
    def get_text(self, batch_text):
        encoded = self.tokenizer.batch_encode_plus(
            batch_text, return_tensors='pt', padding=self.text_pad_type, 
            max_length=self.max_len, truncation=True,
        )

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

        text_embed = torch.where(
            text_mask, # b,1,1
            text_embed, # b,l,c
            self.null_text_embed, # 1,l,c
        )
        return text_embed 

    def shared_step(self, batch, **kwargs):
        x = self.get_input(batch, self.first_stage_key)
        # text model embed as context 
        context = [ self.get_text(batch[k]) for k in self.context_keys]
        # dropout for classifier free guidance
        drop_p = kwargs['cond_drop_prob'] if 'cond_drop_prob' in kwargs else self.cond_drop_prob
        context = [ self.get_text_token(c, drop_p) for c in context]

        loss, loss_dict = self(x, c_crossattn = context, valid_mask=batch['valid_mask'])
        return loss, loss_dict 

    # redefine p_mean_variance here ! since classifier free guidance is needed ! 
    def p_mean_variance(self, x, t, clip_denoised:bool, *args, **kwargs):
        cond_scale = kwargs['cond_scale'] if 'cond_scale' in kwargs else 1.0
        model_out = self.model(x, t, c_crossattn=kwargs['c_crossattn'])
        
        if cond_scale > 1.0:
            uncond_out = self.model(x, t, c_crossattn=kwargs['c_crossattn_null'])
            model_out = cond_scale * (model_out - uncond_out) + uncond_out

        if self.parameterization == 'eps':
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == 'x0':
            x_recon = model_out 
        else:
            raise ValueError(f'invaid parametrization {self.parameterization:s} for p process !')

        if clip_denoised:
            s = 1.0
            if self.clip_method == 'dynamic':
                s = dynamic_threshold(x_recon, self.dynamic_percentile)

            if isinstance(s, torch.Tensor): #and torch.__version__ < '1.10':
                batch_size = x_recon.size(0)
                x_recon = [ x_recon[idx].clamp(-float(s[idx]),float(s[idx]))/float(s[idx]) for idx in range(batch_size) ]
                x_recon = torch.stack(x_recon, dim=0)
            else:
                x_recon = x_recon.clamp(-s, s) / s
            
        
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance 
    
    @torch.no_grad()
    def sample(self, batch_text, batch_size=16, return_intermediates=True, *args, **kwargs):
        batch_size = len(batch_text)
        image_size = self.image_size
        channels = self.channels 

        context = [ self.get_text( batch_text ) ]
        cond_context  = [self.get_text_token(c, 0.0) for c in context]
        uncond_context = [self.get_text_token(c, 1.0) for c in context]

        return self.p_sample_loop(
            (batch_size, channels, image_size, image_size), 
            return_intermediates=return_intermediates, 
            c_crossattn = cond_context,
            c_crossattn_null = uncond_context,
            cond_scale = self.cond_scale,
        )

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=8, sample=True, return_keys=None,  split='train', *args, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        batch_text = batch[self.context_keys[0]] # currently only one context key
        if 'valid_mask' in batch:
            valid_mask = batch['valid_mask']
            x = x[valid_mask == 1, ...]
            batch_text = [batch_text[i] for i in range(len(batch_text)) if valid_mask[i] > 0.]

        batch_size = x.size(0)
        if batch_size == 0: return log 

        N = min(batch_size, N)
        n_row = min(batch_size, n_row)
        x = x[:N, ...].to(self.device)
        batch_text = batch_text[:N]

        log["inputs"] = x 
        log["texts"] = batch_text

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
                samples, denoise_row = self.sample(batch_text, return_intermediates=True, *args, **kwargs)
                log['samples'] = samples 
                log['denoise_row'] = self._get_rows_from_list(denoise_row)
        
        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}

        return log 
        