import os, sys, argparse
from os.path import dirname, abspath, realpath, join
base_dir = dirname(dirname(realpath(__file__)))
sys.path.insert(0, base_dir)

import pyarrow 
import pyarrow.parquet as pq 
import pandas as pd 
import io
from PIL import Image 
import random 

from utils.hdfs_utils import hdfs_ls, hdfs_put 
from utils.utils import im2byte, mkdirs, parse_tags, parse_remark

import numpy as np
import torch 
from torch.utils.data import IterableDataset, DataLoader, random_split
import torchvision as tv 
from torchvision import transforms

import pytorch_lightning as pl 
from transformers import AutoTokenizer, T5Tokenizer

# harry table part 
import harrytable 
from harrytable.fileutil import fetch_hdfs_cached 
from harrytable.df2tensor import IDF2Tensor 
from harrytable.loader import TableLoader, FakeEpochLoader
from harrytable.tbase import TConfig 
from harrytable.sbase import StreamConfig 
from harrytable.tstream import LongRunTStream

def process(imbytes, size=None):
    buffer = io.BytesIO(imbytes)
    image = Image.open(buffer).convert('RGB')
    if size is not None:
        image = image.resize((size, size), resample=Image.Resampling.BICUBIC)
    image = np.array(image).astype(np.uint8)
    image = (image / 127.5 - 1.0).astype(np.float32)
    return image

class Image2Tensor(IDF2Tensor):
    def __init__(self, image_key, size=256, transform=None):
        self.size = size
        norm = transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )

        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
                norm
            ])
        
        self.transform = transform
        self.image_key = image_key 
        self.output_fields = ['image']

    def __call__(self, df):
        im_list = df[self.image_key].apply(lambda im:process(im, self.size)).to_list()
        im_list = [torch.from_numpy(i) for i in im_list] # size b list of (h,w,ch)
        # tensor_list = pil_col.apply(self.transform).to_list()
        # im_tensor = torch.stack(tensor_list, dim=0)
        return {'image': im_list}

class Text2Tensor(IDF2Tensor):
    def __init__(self, text_key, tokenizer_pth=None):
        self.text_key = text_key
        self.tokenizer = None 
        self.output_fields = ['c_crossattn', 'valid_mask']
        if tokenizer_pth is not None:
            # self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_pth)
            # self.output_fields = ['input_ids', 'token_type_ids', 'attention_mask']
            self.output_fields = ['input_ids', 'attention_mask']
    
    def __call__(self, df):
        text_list = df[self.text_key].to_list()
        out_list = []
        for t in text_list:
            if self.text_key == 'keywords':
                chn_tags, eng_tags = parse_tags(t) 
                out_list.append(' '.join(chn_tags + eng_tags))
            elif self.text_key == 'remark':
                chn_remarks, eng_remarks = parse_remark(t)
                out_list.append( chn_remarks )
            
        if self.tokenizer is not None:
            tok_dict = self.tokenizer(
                text_list, 
                padding='max_length',
                max_length=256, truncation=True
            )
            return {k: torch.LongTensor(v) for k,v in tok_dict.items()}
        else: 
            # weight is 0 for blank text items 
            valid_list = [0 if txt == '' else 1 for txt in out_list] 
            valid_list = torch.FloatTensor(valid_list)
            return {'c_crossattn': out_list, 'valid_mask': valid_list}

class MergePreprocess(IDF2Tensor):
    def __init__(self, im_df2tensor, text_df2tensor):
        im_fields = im_df2tensor.output_fields 
        text_fields = text_df2tensor.output_fields 
        assert 'image' in im_fields 
        self.output_fields = im_fields + text_fields
        self.fn1 = im_df2tensor
        self.fn2 = text_df2tensor 

    def __call__(self, df):
        im_tdict = self.fn1(df)
        text_tdict = self.fn2(df)
        return {**im_tdict, **text_tdict}
        # try to filter out items with blank text 
        # merge_df = pd.DataFrame({**im_tdict, **text_tdict})
        # mask = merge_df['c_crossattn'].isin(['', ' '])
        # return merge_df[~mask].to_dict('list')

class Text2ImageDataLoader(pl.LightningDataModule):
    def __init__(self, batch_size, train_hdfs_root, test_hdfs_root=None, val_hdfs_root=None,\
            shuffle=False, num_workers=None, size=64, im_key='imbytes', text_key='keywords', \
            flip_p=0.5, filter=False, tokenizer_pth=None, num_epoch=1):
        super().__init__()
        self.num_epoch = num_epoch
        self.batch_size = batch_size 
        self.train_hdfs_root = train_hdfs_root 
        self.test_hdfs_root = test_hdfs_root 
        self.val_hdfs_root = val_hdfs_root 
        self.shuffle = shuffle 
        self.size = size 
        self.num_workers = num_workers if num_workers is not None else batch_size * 2

        self.im_key = im_key 
        self.text_key = text_key
        assert self.text_key in ['keywords', 'remark']
        self.filter = filter 
        self.flip_p = flip_p 

        self.df2tensor = MergePreprocess(
            Image2Tensor('imbytes', size=64), 
            Text2Tensor(self.text_key, tokenizer_pth=None)
        )
        self.train_loader = None 
        self.test_loader = None 
        self.predict_loader = None 
        self.datasets = {}

    def prepare_data(self):
        pass 

    def build_dloader(self, stage, hdfs_root):
        if stage == 'fit': ratio = 1.0
        else: ratio = 0.1

        tconfig = TConfig(hdfs_dir=hdfs_root, frac=ratio, )
        tloader = TableLoader(
            shuffle = True if stage == 'fit' else False, 
            drop_last = True if stage == 'fit' else False,
            num_worker=self.num_workers, 
            array_batch_size=self.batch_size, 
            array_size=1, 
        )

        sconfig = StreamConfig(
            stream_name='text2im_{}'.format(stage), 
            num_epoch=self.num_epoch,
            is_mmap=True
        )
        tstream = LongRunTStream(sconfig, tconfig, 0)
        
        dloader = tloader.load_tstream(tstream, self.df2tensor)
        return dloader 


    def setup(self, stage=None):
        print('preparing setting up data {}'.format(self.train_hdfs_root))
        if stage is None or stage == 'fit':
            self.train_loader = self.build_dloader('fit', self.train_hdfs_root)
            # self.datasets['fit'] = self.train_loader 
        if stage is None or stage == 'test':
            self.test_loader = self.build_dloader('test', self.test_hdfs_root)
        if stage is None or stage == 'predict':
            self.predict_laoder = self.build_dloader('predict', self.val_hdfs_root)

    def train_dataloader(self):
        return FakeEpochLoader(self.train_loader, epoch_len=1000000)

    def val_dataloader(self):
        return FakeEpochLoader(self.test_loader, epoch_len=100)

    def test_dataloader(self):
        return FakeEpochLoader(self.predict_dataloader, epoch_len=100)