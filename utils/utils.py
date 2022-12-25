import os 
from os.path import join, isdir, isfile, exists 
import copy 
import numpy as np 
import cv2 
import pyarrow as pa 
import pandas as pd 
from collections import defaultdict, OrderedDict
from matplotlib import pyplot as plt 
import torchvision.utils as vutils
from PIL import Image   
import io 
import torch
import mplfonts 
import math 
import ray 

class DB_Saver(object):
    def __init__(self, keys, save_freq, save_dir):
        self.keys = keys 
        # self.dbs = dict(zip(keys, [[]] * len(keys)))
        # self.dbs = defaultdict(list)
        self.dbs = {}
        self.reset()

        self.save_freq = save_freq 
        self.save_dir = save_dir

    def reset(self):
        self.len = 0 
        for k in self.keys:
            self.dbs[k] = [] 

    def save(self):
        print('build db table complete & start to write parquet !')
        dbs_tab = ray.data.from_arrow( pa.table(self.dbs) )
        # dbs_tab = ray.data.from_pandas( pd.DataFrame( self.dbs))
        print('new db table count {:d} with len {:d} !'.format(dbs_tab.count(), self.len))
        dbs_tab.write_parquet(self.save_dir)
        self.reset()


    def append(self, **kwargs):
        for k,v in kwargs.items():
            if k not in self.dbs:
                raise ValueError('invalid key {}'.format(k))

            if isinstance(v, list):
                self.dbs[k].extend(v)
            else: 
                self.dbs[k].append(v)
            
        cur_len = 1 
        if isinstance(kwargs[self.keys[0]], list):
            cur_len = len(kwargs[self.keys[0]])

        self.len += cur_len

    def __len__(self):
        return self.len 
    
def is_chn(s):
    if '\u4e00' <= s <= '\u9fa5':
        return True 
    else:
        return False

def is_eng(s):
    sord = ord(s)
    if 97 <= sord <= 122 or 65 <= sord <= 90:
        return True  
    else:
        return False 

def parse_remark(text):
    chn_lid, chn_rid = len(text), 0
    eng_lid, eng_rid = len(text), 0
    for idx, s in enumerate(text):
        if is_eng(s): 
            eng_lid = min(eng_lid, idx)
            eng_rid = max(eng_rid, idx)
        elif is_chn(s):
            chn_lid = min(chn_lid, idx)
            chn_rid = max(chn_rid, idx)
    if eng_lid <= chn_rid: eng_lid = chn_rid+1

    return text[chn_lid:chn_rid+1].strip(), text[eng_lid:eng_rid+1].strip()
    
def parse_tags(tags):
    tags = tags.split(',')
    chn_tags = []
    eng_tags = []
    for t in tags:
        chn_idx = -1 
        for idx, s in enumerate(t):
            if is_chn(s): 
                chn_idx = idx 
                break 
        eng_idx = -1 
        for idx, s in enumerate(t):
            if is_eng(s):
                eng_idx = idx 
                break  
        if chn_idx != -1 and eng_idx != -1:
            if chn_idx < eng_idx:
                chn_s, eng_s = t[:eng_idx], t[eng_idx:]
            else: 
                eng_s, chn_s = t[:chn_idx], t[chn_idx:]
            chn_tags.append(chn_s)
            eng_tags.append(eng_s)
        elif chn_idx != -1:
            chn_tags.append(t)
        elif eng_idx != -1:
            eng_tags.append(t) 
    
    # chn_len = len(chn_tags)
    # eng_len = len(eng_tags)
    # if chn_len != eng_len: 
    #     info = 'len of chn {:d} != eng {:d}'.format(len(chn_tags), len(eng_tags))
    #     print(info)

    return chn_tags, eng_tags 

def mkdirs(dpth):
    if not exists(dpth):
        os.makedirs(dpth)

def get_hdfs_subset(hdfs_root, suffix='.parquet'):
	result = os.popen(f'hdfs dfs -ls {hdfs_root:s}')
	info_str = result.read()

	files = []
	for row in info_str.split('\n'):
		for s in row.split(' '):
			if s.endswith(suffix):
				files.append(s)
	return files 

def get_fname(fpth):
    out = fpth.split('/')[-1]
    out = out.split('.')[0]
    return out 

def im2byte(img):
	if isinstance(img, np.ndarray):
		img = Image.fromarray(img)
	img_format = 'PNG' if img.format is None else img.format
	img_byte = io.BytesIO()
	img.save(img_byte, format=img_format)
	img_byte = img_byte.getvalue()
	return img_byte

def tensor2byte(t, dtype=np.float32):
    if isinstance(t, torch.Tensor):
        t = t.cpu().numpy()
    return t.astype(dtype).tobytes() # decoded into 1-d array

def byte2tensor(b, shape, dtype=np.float32):
    assert isinstance(shape, (list, tuple))
    b = np.frombuffer(b, dtype=dtype)
    b = torch.from_numpy(b).view(*shape)
    return b 

# process PIL image to numpy array 
def process_image(image, size=None, dtype=np.float32):
	assert dtype in [np.float32, np.uint8]
	if size is not None:
		image = image.resize((size, size))

	image = np.array(image).astype(np.uint8) #0-255
	if dtype == np.float32:
		image = (image / 127.5 -1.0).astype(np.float32)
	return image

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

def norm_each(x, eps=1e-12):
    '''x: b,c,h,w '''
    bsize = x.size(0)
    xmin = x.view(bsize, -1).min(dim=-1)[0]
    xmax = x.view(bsize, -1).max(dim=-1)[0]
    xmin = right_pad_dims_to(x, xmin)
    xmax = right_pad_dims_to(x, xmax)
    return (x - xmin) / (xmax - xmin + eps)

def plot_images_with_texts(images, texts, save_pth, ncol=4, fontsize=8, nsplit=8):
    nim = len(images)
    assert len(texts) == nim, \
        'number of image {:d} != number of text {:d} !'.format(nim, len(texts))

    if len(texts) and len(texts[0]) and is_chn(texts[0][0]):
        mplfonts.use_font('SimHei')

    nrow = math.ceil(nim / ncol)
    fig, axes = plt.subplots(nrow, ncol)
    if isinstance(images, torch.Tensor):
        images = custom_to_np(images).cpu().numpy()
    for idx, (im, text) in enumerate(zip(images, texts)):
        x = int(idx / ncol); y = idx % ncol 
        text = [text[i:i+nsplit] for i in range(0, len(text), nsplit)]
        text = '\n'.join(text)
        if nrow == 1 or ncol == 1:
            ax = axes[idx]
        else:
            ax = axes[x, y]
        ax.imshow(im, interpolation='nearest')
        ax.set_title(text, fontsize=fontsize, wrap=True)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.savefig(save_pth, facecolor='white') #,box_inches='tight')


def plot_landmarks(im, landmarks, radius=1, offset=2):
	im = copy.deepcopy(im)
	for idx, (x, y) in enumerate(landmarks):
		cv2.circle(im, (int(x), int(y)), radius, [255, 0, 0], 2*radius)
		cv2.putText(im, str(idx), (int(x)+offset, int(y)+offset), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 255], 1)
	return im 

def plot_box(im, box, thick=2):
	im = copy.deepcopy(im)
	x1, y1, x2, y2 = box
	pt1_list = [(x1, y1), (x2, y1), (x1, y2), (x1, y1)]
	pt2_list = [(x1, y2), (x2, y2), (x2, y2), (x2, y1)]
	for (pt1, pt2) in zip(pt1_list, pt2_list):
		cv2.line(im, pt1, pt2, [0, 255, 0], thick)
	return im

def save_vis(im_list, save_pth=None, title=None):
    n_vis = len(im_list)
    if title is not None and is_chn(title[0]):
        print('set params for chn success')
        # plt.rcParams['font.sans-serif']= ['SimHei']
        # plt.rcParams['axes.unicode_minus'] = False
        mplfonts.use_font('SimHei')
    plt.axis('off') 
    for vid, im in enumerate(im_list):
        plt.subplot(1, n_vis, vid+1)
        plt.imshow(im) 
    if title is not None:
        plt.title(title)
    if save_pth is not None:
        plt.savefig(save_pth)
    plt.close() 