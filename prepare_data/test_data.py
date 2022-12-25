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
from ldm.util import instantiate_from_config
from utils.utils import custom_to_np, custom_to_pil, mkdirs, parse_tags, im2byte, parse_remark, get_hdfs_subset
from utils.hdfs_utils import hdfs_ls, hdfs_mkdirs
from PIL import Image 
import pyarrow as pa 
from tqdm import tqdm 

import ray 
os.environ['RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE'] = '1'
MAX_MEM = 1024 * 1024 * 1024 * 32
ray.init(ignore_reinit_error=True, _memory=MAX_MEM, _driver_object_store_memory=MAX_MEM, object_store_memory=MAX_MEM, _redis_max_memory=MAX_MEM)
yzh_hdfs='hdfs://haruna/home/byte_ailab_vc_video_summarization/user/yinzihao'

def toy_ray_filter_example():
    # ds = ray.data.range(10000)  
    # ds.count() 
    # ds = ds.map_batches(lambda batch: [x for x in batch if x % 2 == 0])  

    data = ray.data.range(10000)
    print(data.count())
    data = data.map_batches(lambda batch: [x for x in batch if x % 2 == 0], batch_size=4096)
    print(data.count())
    for batch in data.iter_batches(batch_size=10):
        print([x for x in batch])
        import pdb; pdb.set_trace()


def size_filter_fn(batch, im_key='_c0', tgt_size=256): #'_c0'):
    # print(type(x))
    outs = {im_key: []}
    for x in batch[im_key]:
        if not isinstance(x,  bytes): x = x.as_py()
        image = Image.open(io.BytesIO(x))
        h, w = image.size 
        if h >= tgt_size and w >= tgt_size: 
            outs[im_key].append(x)
    return pa.table(outs)

def text_filter_fn(batch, im_key='imbytes', tgt_size=64):
    outs = {im_key: [], 'keywords': [], 'remark': []}
    if not isinstance(batch, pd.DataFrame): batch = batch.to_pandas()
    for ridx, row in batch.iterrows():
        chn_tags, _ = parse_tags(row['keywords'])
        if len(chn_tags) == 0: continue 
        chn_remark, _  = parse_remark(row['remark'])
        if len(chn_remark) == 0: continue 

        chn_tags = ' '.join([str(c) for c in chn_tags])
        outs['keywords'].append(chn_tags)
        outs['remark'].append(chn_remark)

        x = row['imbytes']
        if not isinstance(x, bytes): x = x.as_py()
        x = Image.open(io.BytesIO(x))
        x = x.resize((tgt_size, tgt_size))
        outs[im_key].append( im2byte(x) )

    return pa.table(outs)

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
    parser.add_argument('--data', type=str, default='background', help='test data name')
    parser.add_argument('--base', type=str, default='', help='base config')
    parser.add_argument('--im_key', type=str, default='_c0', help='image key in hdfs')
    parser.add_argument('--tgt_size', type=int, default=256, help='tgt size for filter')
    parser.add_argument('--dst_hpth', type=str, \
		default='hdfs://haruna/home/byte_ailab_vc_video_summarization/user/yinzihao/data', 
		help='dst dir to save parquet'
	)
    parser.add_argument('--filter_method', type=str, default='size', choices=['size', 'text'])
    opt, unknown = parser.parse_known_args()
    assert opt.data in ['face', 'background', 'text2image']

    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.load(opt.base)
    config = OmegaConf.merge(config, cli)
    print(config.data)
    
    # data = instantiate_from_config(config.data)
    # data.prepare_data()
    # data.setup()

    # print("#### Data #####")
    # for k in data.datasets:
    #     print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    # l1 = []; l2 = []
    # for example in data.train_set.iter_rows():
    #     keywords = example['keywords']
    #     remark = example['remark']
    #     l1.append(len(keywords))
    #     l2.append(len(remark))
    # print(max(l1), min(l1), sum(l1) / float(len(l1)))
    # print(max(l2), min(l2), sum(l2) / float(len(l2)))
    # assert 0

    # hdir = 'hdfs://haruna/home/byte_ailab_vc_video_summarization/user/yinzihao/data/face_sr_256'
    # hdir = join(yzh_hdfs, 'data/tuchong/background_sr_256')
    # hdir = join(yzh_hdfs, 'data/tuchong/tuchong_background')
    # hdir = join(yzh_hdfs, 'data/tuchong/image_64_chn_keywords_remark')
    # pq_pths = hdfs_ls(hdir)
    # print('{} has {} parquets'.format(hdir, len(pq_pths)))
    # ds = ray.data.read_parquet(hdir)
    # print('len of {} is {}'.format(0, ds.count()))
    # cnt = 0
    # for row in tqdm(ds.iter_rows(), position=0, leave=True):
    #     x = row[opt.im_key]
    #     # if not isinstance(x, bytes): x = x.as_py()
    #     try:
    #         im = Image.open(io.BytesIO(x))
    #     except:
    #         continue 
    #     keywords = row['keywords']
    #     remark = row['remark']
    #     im.resize((256, 256))
    #     im.save(f'./{cnt}_demo.png')
    #     # print(keywords)
    #     # print('#' * 50)
    #     print(remark)
    #     print('#' * 50)
    #     cnt += 1 
    #     if cnt >= 9: break 
    # assert 0
        
    hdfs_mkdirs(opt.dst_hpth)
    # toy_ray_filter_example()
    
    
    # eval(f'test_{opt.data}_data')(opt, config.data)
    filter_data(opt, config.data, opt.filter_method)