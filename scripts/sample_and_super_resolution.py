import argparse, os, sys, glob, datetime, yaml
from os.path import exists, isdir, isfile, join, dirname, realpath

sys.path.insert(0, dirname(realpath(__file__)))
import omegaconf 
import torch 
import torch.nn.functional as F  
import time
import numpy as np 
from glob import glob 
from omegaconf import OmegaConf     
from PIL import Image 
from tqdm import tqdm, trange
from collections import defaultdict 

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.image_degradation.bsrgan import add_blur  
from ldm.util import instantiate_from_config
from utils.utils import mkdirs, custom_to_np, custom_to_pil, norm_each

rescale = lambda x: (x + 1.) / 2.

def get_parser():
    parser = argparse.ArgumentParser() 
    parser.add_argument('-r', '--resume', type=str, default='', help='generation checkpoint logdir')
    parser.add_argument('--sr_resume', type=str, default='', help='super-resolution checkpoint logdir')
    parser.add_argument('-n', '--n_samples', type=int, default=50000, nargs='?', help='number of samples to draw')
    parser.add_argument('-c', '--custom_steps', type=int, default=50, nargs='?', help='number of steps for ddim & fastdpm fsampling')
    parser.add_argument('--batch_size', type=int, default=50, nargs='?', help='the bs')
    parser.add_argument('-v', '--vanilla_sample', default=False, action='store_true', )
    parser.add_argument('--config_name', type=str, default='', help='to specify config name')
    parser.add_argument('--logdir', type=str, default='./logs', help='result log dir')
    return parser 

def save_logs(logs, path, n_saved=0, key="sample", np_path=None, prefix="base", tgt_size=256):
    path = join(path, prefix)
    mkdirs(path)
    for k in logs:
        if k == key:
            batch = logs[key]
            if np_path is None:
                for x in batch:
                    img = custom_to_pil(x)
                    h, w = img.size 
                    if h != tgt_size or w != tgt_size:
                        img = img.resize((tgt_size, tgt_size))
                    imgpath = os.path.join(path, f"{key}_{n_saved:06}.png")
                    img.save(imgpath)
                    n_saved += 1
            else:
                npbatch = custom_to_np(batch)
                shape_str = "x".join([str(x) for x in npbatch.shape])
                nppath = os.path.join(np_path, f"{n_saved}-{shape_str}-samples.npz")
                np.savez(nppath, npbatch)
                n_saved += npbatch.shape[0]
    return n_saved

def get_info_from_resume(resume):
    if not exists(resume):
        raise ValueError('Cannot find  {}'.format(resume))
    if isfile(resume):
        # logdir/ checkpoints / model.pt
        logdir = resume.rsplit('/', 2) 
        if logdir[1] != 'checkpoints':
            logdir = join(logdir[0], logdir[1])
        else: 
            logdir = logdir[0]

        print(f'Logdir is {logdir}')
        ckpt = resume 
    elif isdir(resume):
        logdir = resume.rstrip("/")
        ckpt = join(logdir, 'last.ckpt')
    else:
        raise ValueError(f'{resume} is neither file or dir')
    return logdir, ckpt

def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    # model.eval() 
    return model 

def load_model(model_config, ckpt, gpu, eval_mode):
    if ckpt: 
        print(f'Loading model from {ckpt}')
        pl_sd = torch.load(ckpt, map_location='cpu')
        global_step = pl_sd['global_step']
    else:
        # raise ValueError(f'load nothing into model !')
        pl_sd = {"state_dict": None}
        global_step = None  

    model = load_model_from_config(model_config, pl_sd['state_dict'])
    if eval_mode: model.eval()
    else: model.train() 
    return model, global_step

@torch.no_grad()
def sample(model, batch_size=50, vanilla=False, custom_steps=None):
    log = dict() 
    in_channel, im_size = model.model.diffusion_model.in_channels, \
        model.model.diffusion_model.image_size 
    shape = [batch_size, in_channel, im_size, im_size]
    with model.ema_scope('Plotting'):
        t0 = time.time() 
        sample, intermediates = model.p_sample_loop(shape, return_intermediates=True)
        t1 = time.time() 
    log['sample'] = sample 
    log['time'] = t1 - t0 
    log['throughput'] = sample.shape[0] / (t1 - t0)
    return log 

@torch.no_grad()
def super_resolution(model, cond_log, batch_size=50, vanilla=False, custom_steps=None):
    log = dict()

    in_channel, im_size = model.model.diffusion_model.in_channels,\
        model.model.diffusion_model.image_size

    cond_im = cond_log['sample'] # b,c,h,w
    cond_im = norm_each(cond_im) * 2.0 - 1.0
    # cond_im = torch.clamp(cond_im, -1.0, 1.0)
    # cond_im = custom_to_np( cond_im )
    # cond_im = add_blur(cond_im, sf=4)
    # cond_im = torch.from_numpy(cond_im / 127.5 - 1.0).float() 

    shape = [batch_size, in_channel - cond_im.size(1), im_size, im_size]
    if cond_im.size(2) != im_size or cond_im.size(3) != im_size:
        cond_im = F.interpolate(cond_im, size=shape[2:], mode='bicubic') #,align_corners=True)
    
    context = [ cond_im ]
    with model.ema_scope('Plotting'):
        t0 = time.time()
        sample, intermediates = model.p_sample_loop(shape, return_intermediates=True, c_concat=context)
        t1 = time.time()
    log['sample'] = sample
    log['time'] = t1 - t0 
    log['throughput'] = sample.shape[0] / (t1 - t0)
    return log 

def run(models_dict, logdir, batch_size=50, vanilla=False, n_samples=50000, custom_steps=None, nplog=None):
    tstart = time.time()
    pre_n_saved = len(glob(join(logdir, '*.png'))) - 1 
    
    images_dict = defaultdict(list)
    print(f'Running unconditional samping for {n_samples} samples')
    for _ in trange(n_samples // batch_size, desc='sampling batches (unconditional)'):
        base_logs = sample(models_dict['base'], batch_size=batch_size, custom_steps=custom_steps)
        n_saved = save_logs(base_logs, logdir, n_saved=pre_n_saved, key='sample', prefix='base')
        images_dict['base'].extend([ custom_to_np(base_logs['sample']) ])

        sr_logs = super_resolution(models_dict['sr'], base_logs, batch_size=batch_size,)
        n_saved = save_logs(sr_logs, logdir, n_saved=pre_n_saved, key='sample', prefix='sr')
        images_dict['sr'].extend([ custom_to_np(sr_logs['sample']) ])

        if n_saved >= n_samples:
            print(f'Finish after generating {n_saved} samples')
            break 
        pre_n_saved = n_saved


if __name__ == '__main__':
    now = datetime.datetime.now().strftime('')
    sys.path.append( os.getcwd() )
    command = " ".join(sys.argv)

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    ckpt = None

    all_infos = []
    if not opt.resume:
        base_logdir, base_ckpt = None, None 
    else:
        base_logdir, base_ckpt = get_info_from_resume(opt.resume)
        all_infos.append( ('base', base_logdir, base_ckpt) )
    
    if not opt.sr_resume:
        sr_logdir, sr_ckpt = None, None 
    else:
        sr_logdir, sr_ckpt = get_info_from_resume(opt.sr_resume)
        all_infos.append( ('sr', sr_logdir, sr_ckpt) )

    # base_configs = sorted(glob(join(logdir, 'configs', '*.yaml')))

    gpu = True 
    eval_mode = False #True 

    cli = OmegaConf.from_dotlist(unknown)
    models_dict = {}
    for (name, logdir, ckpt_pth) in all_infos:
        cfg_pth = join(logdir, 'configs', 'config.yaml')
        config = OmegaConf.load(cfg_pth)
        config = OmegaConf.merge(config, cli)
        model, global_step = load_model(config.model, ckpt_pth, gpu, eval_mode)
        print(f'global_step: {global_step}')
        print(config.model)
        print(75 * '=')
        models_dict[name] = model

    print('logging to:')
    
    logdir = join(opt.logdir, 'samples', f'{global_step:08}', now)
    imglogdir = join(logdir, 'img')
    numpylogdir = join(logdir, 'numpy')
    mkdirs(imglogdir)
    mkdirs(numpylogdir)
    print(logdir)
    print(75 * '=')

    # write config out 
    sampling_file = join(logdir, 'samping_config.yaml')
    sampling_conf = vars(opt)

    with open(sampling_file, 'w') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)
    print(sampling_conf)

    run(models_dict, imglogdir, n_samples=opt.n_samples, \
        batch_size=opt.batch_size, custom_steps=opt.custom_steps)
    print('done.')

