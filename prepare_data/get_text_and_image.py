import argparse, sys, os
from os.path import join, isdir, isfile, exists, dirname, realpath 
base_pth = dirname(realpath(__file__))
sys.path.insert(0, base_pth)

# add ray part 
os.environ['RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE'] = '1'
import ray 
MAX_MEM = (1024 * 1024 * 1024) * 64

ray.init(ignore_reinit_error=True, _memory=MAX_MEM, object_store_memory=MAX_MEM, _redis_max_memory=MAX_MEM)

import numpy as np 
import torch 
from PIL import Image  
import io
from tqdm import tqdm 
import random  
import pyarrow as pa 


from utils.hdfs_utils import hdfs_mkdirs, hdfs_put, hdfs_ls
from utils.utils import get_hdfs_subset, save_vis, get_fname, im2byte, \
    process_image, mkdirs, is_chn, is_eng, parse_tags
from matplotlib import pyplot as plt 


base_dir='/mnt/bd/yinzihaodata'
code_dir=join(base_dir, 'code')
data_dir=join(base_dir, 'data')
pretrained_dir=join(base_dir, 'pretrained_models')
        
def is_background(chn_tags, eng_tags, keywords=[u'背景', u'插图', u'图案', u'设计']):
    if len(chn_tags) == 0: return False
    flag = False 
    for t in chn_tags:
        if u'人' in t and t != u'没有人': 
            return False 
        if t in [u'风景', u'自然', u'景观', u'动物', u'建筑', u'植物', u'美食', u'食物']:
            return False
        if t.strip() in keywords:
            flag = True
    return flag

class DB_Saver(object):
    def __init__(self, keys, save_freq, save_dir):
        self.keys = keys 
        self.save_dir = save_dir
        self.save_freq = save_freq 

        self.dbs = {}
        self.len = 0
        self.reset()

    def reset(self):
        self.len = 0 
        for k in self.keys:
            self.dbs[k] = [] 

    def save(self):
        print('build db table complete & start to write parquet !')
        # dbs_tab = pa.table(dbs)
        dbs_tab= ray.data.from_arrow( pa.table(self.dbs) )
        print('new db table count {:d} !'.format(dbs_tab.count()))
        # import pdb; pdb.set_trace()
        dbs_tab.write_parquet(self.save_dir)
        self.reset()

    def append(self, **kwargs):
        for k,v in kwargs.items():
            if k in self.dbs:
                self.dbs[k].append(v)
            else: 
                raise ValueError('invalid key {}'.format(k))
        self.len += 1 

    def __len__(self):
        return self.len
    
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='argument parser for face alignment')
    parser.add_argument('--size', type=int, default=256, help='tgt image size')
    parser.add_argument('--src_hpth', type=str, required=True, help='src hdfs data path')
    parser.add_argument('--dst_hpth', type=str, \
        default='hdfs://haruna/home/byte_ailab_vc_video_summarization/user/yinzihao/data', 
        help='dst dir to save parquet'
    )
    parser.add_argument('--suffix', type=str, default='.parquet', help='suffix of files')
    parser.add_argument('--verbose', action='store_true', help='whether to print extra info')
    parser.add_argument('--save_name', type=str, required=True, help='should specify hdfs root name')
    parser.add_argument('--save_freq', type=int, default=50000, help='save every rows of data')
    args = parser.parse_args() 
    
    save_dir = join(args.dst_hpth, args.save_name)
    hdfs_mkdirs(save_dir)

    cw_dir = os.getcwd()
    print(f'current working dir: {cw_dir}')
    vis_dir = join(cw_dir, 'vis', args.save_name)
    mkdirs(vis_dir)

    # subset_list = get_hdfs_subset(args.src_hpth, suffix='.parquet')
    subset_list = hdfs_ls(args.src_hpth)
    subset_list = [p for p in subset_list if p.endswith('.parquet')]
    print('{} === HAS ==> {} parquets'.format(args.src_hpth, len(subset_list)))
    
    # subset_list = random.sample(subset_list, 1)
    # some pq has blank keywords !
    # subset_list = [join(args.src_hpth, 'part-49274-066d7df5-086c-4af6-bfc4-97aaacd7a1bf-c000.gz.parquet')]

    # dbs_saver = DB_Saver(['image', 'chn_tags', 'eng_tags'], args.save_freq, save_dir)
    dbs_saver = DB_Saver(['image', 'chn_tags'], args.save_freq, save_dir)
    vis_freq =  max(1, len(subset_list) // 50 )
    for sidx, subset_pth in enumerate(subset_list):
        if sidx < 1628: continue
        dset = ray.data.read_parquet(subset_pth)
        dset_len = dset.count()
        print('{} dset length: {}'.format(sidx, dset_len))
        cur_fname = get_fname(subset_pth)
        # cur_vis_dir = join(vis_dir, get_fname(subset_pth))
        # mkdirs(cur_vis_dir)
        
        npre = len(dbs_saver)

        for idx, data in tqdm(enumerate(dset.iter_rows())):
            chn_tags, eng_tags = parse_tags(data['keywords'])
            flag = is_background(chn_tags, eng_tags)
            if not flag: 
                continue 

            image_id = data['image_id']
            im_bytes = data['imbytes']
            im = Image.open(io.BytesIO(im_bytes)).convert('RGB')
            h, w = im.size 
            if h < args.size and w < args.size:
                continue 

            # dbs_saver.append(image=im_bytes, chn_tags=','.join(chn_tags), eng_tags=','.join(eng_tags))
            dbs_saver.append(image=im_bytes, chn_tags=','.join(chn_tags))

        if sidx % vis_freq == 0:
            im_ori = Image.open(io.BytesIO(im_bytes))            
            im_npy = process_image(im_ori, size=None, dtype=np.uint8) 
        
            # if args.verbose:
                # print(idx, image_id, data['keywords'], eng_tags, chn_tags, im_npy.shape)
            tags_all = ','.join([str(t) for t in chn_tags])
            save_pth = join(vis_dir, '{}_{:d}_{:s}.jpg'.format(cur_fname, sidx, tags_all[:16]))
            save_vis([ im_npy ], save_pth, title=None) #tags_all[:32])

        npost = len(dbs_saver)
        print('db len pre {:d} -> {:d}'.format(npre, npost))
        if len(dbs_saver) >= args.save_freq:
            dbs_saver.save()

