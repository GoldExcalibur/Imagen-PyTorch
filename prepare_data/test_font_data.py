import os, sys, argparse
from os.path import join, isdir, exists, isdir, isfile, dirname, realpath
base=dirname(dirname(realpath(__file__)))
sys.path.insert(0, base)

from omegaconf import OmegaConf
import numpy as np 
import cv2 
import io 
import pandas as pd 

from ldm.data.ray_reader import RayDataLoader
from ldm.data.font import FontDataLoader
from ldm.util import instantiate_from_config
from utils.utils import custom_to_np, custom_to_pil, mkdirs, parse_tags, im2byte, parse_remark, get_hdfs_subset
from PIL import Image 
from tqdm import tqdm 

# import ray 
# os.environ['RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE'] = '1'
# MAX_MEM = 1024 * 1024 * 1024 * 32
# ray.init(ignore_reinit_error=True, _memory=MAX_MEM, _driver_object_store_memory=MAX_MEM, object_store_memory=MAX_MEM, _redis_max_memory=MAX_MEM)
yzh_hdfs='hdfs://haruna/home/byte_ailab_vc_video_summarization/user/yinzihao'

def filter_data(opt, data_config, filter_method):
    # data = instantiate_from_config(data_config)
    # data.prepare_data()
    # data.setup(stage='test')

    # save_dir = join(base, 'logs', 'images_real256')
    # mkdirs(save_dir)


    # for idx, example in enumerate(data.test_ds):
    #     im = example['image']
    #     im = (im + 1) * 127.5
    #     im = np.clip(im, 0, 255).astype(np.uint8)
    #     h, w, c = im.shape
    #     cv2.imwrite(join(save_dir, f'{h}_{idx}.png'), im)
    #     if idx == 10: break 

    hroot = data_config.params.train_hdfs_root
    # total_pq_pths = hdfs_ls(hroot)
    # total_pq_pths = [p for p in total_pq_pths if p.endswith('.parquet')]
    total_pq_pths = get_hdfs_subset(hroot)
    # pq_pths = data_config.params.val_hdfs_root 
    print('{} has {} parquets !'.format(hroot, len(total_pq_pths)))

    # n = 8
    ntotal = 0
    # pq_pths_slices = [total_pq_pths[i:i+n] for i in range(0, len(total_pq_pths), n)]
    for idx, pq_pths in enumerate(total_pq_pths):
        if idx < 6583: continue 
        ds = ray.data.read_parquet(pq_pths)

        npre = ds.count()
        ds = ds.map_batches(
            lambda x: text_filter_fn(x, opt.im_key, opt.tgt_size), 
            batch_size=4096,
        )
        npost = ds.count()
        if npost < 1000: continue 
        ds.write_parquet(opt.dst_hpth)
        ntotal += npost
        print('{} pre {} post {} total {}'.format(idx, npre, npost, ntotal))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser for test data')
    parser.add_argument('--data', type=str, default='font', help='test data name')
    parser.add_argument('--base', type=str, \
        default='/mnt/bd/yinzihaodata/code/latent-diffusion/configs/ddpm/font.yaml', 
        help='base config'
    )
    parser.add_argument('--tgt_size', type=int, default=80, help='tgt size for filter')
    parser.add_argument('--dst_hpth', type=str, \
		default='hdfs://haruna/home/byte_ailab_vc_video_summarization/user/yinzihao/data', 
		help='dst dir to save parquet'
	)
    opt, unknown = parser.parse_known_args()
    assert opt.data in ['font', 'face', 'background', 'text2image']

    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.load(opt.base)
    config = OmegaConf.merge(config, cli)
    print('data config {}'.format(config.data))
    
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup(stage='fit')

    print("#### Data #####")
    for k in data.datasets:
        print("{}, {}, {:d}".format(
            k, type(data.datasets[k]), len(data.datasets[k])
        ))
    
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    print(len(train_loader), len(val_loader))

    vis_dir = join(base, 'vis', 'font')
    mkdirs(vis_dir)
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 10: break 
        im = batch['image']
        style_cls = batch['style']
        content_cls = batch['content']

        if batch_idx == 0: 
            import pdb; pdb.set_trace()
            
        im_pil = custom_to_pil(im[0]).convert('RGB')
        print(im_pil.size)

        save_pth = join(vis_dir, '{}_{}_{}.png'.format(batch_idx, style_cls[0], content_cls[0]))
        im_pil.save(save_pth)
