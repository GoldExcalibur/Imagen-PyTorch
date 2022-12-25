import os 
from os.path import join, exists, isdir, isfile, dirname 
import numpy as np 
import PIL 
from PIL import Image 
from torch.utils.data import Dataset, IterableDataset, random_split
# from torch.utils.data.dataloader import DataLoader
from torch.utils.data import DataLoader, dataloader 
from torchvision import transforms 

import pytorch_lightning as pl 
import numpy as np 
import random 
import cv2 
import collections 
from collections import defaultdict

from utils.utils import get_fname 

def process(image, size=None):
    if size is not None:
        image = image.resize((size, size))
    image = np.array(image).astype(np.uint8)
    image = (image / 127.5 - 1.0).astype(np.float32)
    return image 

class FontData(Dataset):
    def __init__(self, ds, size=80, is_train=False):
        super().__init__()
        self.ds = ds 
        self.size = size
        self.is_train = is_train 

        if self.is_train:
            random.shuffle(self.ds) 

        # normalize = transforms.Normalize(
        #     mean = [0.5, 0.5, 0.5],
        #     std = [0.5, 0.5, 0.5]
        # )
        # self.transform = transforms.Compose([
        #     transforms.Resize((self.size, self.size)),
        #     transforms.ToTensor(),
        #     normalize
        # ])

        self.prepare_info_dict()

    def prepare_info_dict(self):
        self.style_content_to_dsid = defaultdict(dict)
        for dsid, item in enumerate(self.ds):
            style_id = item['style']
            content_id = item['content']
            self.style_content_to_dsid[style_id][content_id] = dsid 

    # def __iter__(self):
    #     for data in self.ds:
    #         fpath = data['file_name']
    #         # content_id = data['content']
    #         # style_id = data['style']
    #         example = dict()
    #         image = Image.open(fpath).convert('RGB')
    #         example['image'] = process(image, self.size)
    #         yield example 

    def __getitem__(self, index):
        data = self.ds[index]
        fpath = data['file_name']
        # content_id = data['content']
        # style_id = data['style']
        example = dict()
        image = Image.open(fpath).convert('RGB')
        example['image'] = process(image, self.size)
        return example

    def __len__(self):
        return len(self.ds)

    # def __getitem__(self, index):
    #     item = self.ds[index]
    #     # print(type(item), list(item.keys()))
    #     fpath = item['file_name']
    #     style_id = item['style']
    #     content_id = item['content']

    #     image = Image.open(fpath).convert('RGB')
    #     if self.transform:
    #         image = self.transform(image)
    #     example = {}
    #     example['image'] = image #process(image, self.size, self.transform)
    #     example['style'] = style_id
    #     example['content'] = content_id

    #     return example 
         


def has_file_allowed_extension(fname, extensions):
    fname_lower = fname.lower()
    return any(fname_lower.endswith(ext) for ext in extensions)


class FontDataLoader(pl.LightningDataModule):
    def __init__(self, batch_size, data_root, size=80, num_workers=None, \
            test_ratio=0.1, extensions=['png', 'jpg']):
        super().__init__()
        self.batch_size = batch_size 
        self.data_root = data_root 
        self.size = size 
        self.num_workers = num_workers if num_workers is not None else batch_size * 2 # ? 
        self.test_ratio = test_ratio
        self.ds = None 
        self.class_mp = None 
        self.extensions = extensions
        self.datasets = {}

    def prepare_data(self):
        # download or tokenize data called only in main process, 
        # not recomended to set state in here
        pass 
    
    def init_data(self, data_root):
        # data_root: list of subdirs (each subdir is a font class, with all images of this style)
        font_subdirs = os.listdir(data_root) # files & dirs
        font_subdirs = [p for p in font_subdirs if isdir(join(data_root, p)) and 'id' in p]
        # sorted to ensure a consistent order 
        font_subdirs = sorted(font_subdirs, key=lambda f: int(f.split('_')[-1]))
        n_fonts = len(font_subdirs)
        print('==> under {}, we find {:d} fonts'.format(data_root, n_fonts))
        
        self.class_mp = dict(zip( font_subdirs, list(range(n_fonts)) ))
        self.ds = []
        for subdir in font_subdirs:
            cdir = join(data_root, subdir)
            style_idx = self.class_mp[subdir]

            for root, _, fnames in sorted(os.walk(cdir)):
                for fname in sorted(fnames):
                    if has_file_allowed_extension(fname, self.extensions):
                        fpath = join(root, fname)
                        try:
                            content_idx = int(get_fname(fname))
                        except:
                            raise ValueError(f'invalid file name {fname:s} to fetch content id !')
                        item = {'file_name': fpath, 'style': style_idx, 'content': content_idx}
                        self.ds.append( item )

        print('==> dset has {:d} items'.format(len(self.ds)))

    # split data or set datasets (ops on every gpu)
    def setup(self, stage=None):
        print('perparing setting up data: {}'.format(self.ds))
        # init data (get class map dict & dataset list)
        self.init_data(self.data_root) 
        
        self.datasets = dict()
        if stage == 'fit' or stage is None: 
            total_ds = FontData(self.ds, size=self.size, is_train=True)
            ntotal = len(total_ds)
            ntest = int(self.test_ratio * ntotal)
            train_ds, val_ds = random_split(total_ds, [ntotal - ntest, ntest])
            self.datasets['train'] = train_ds 
            self.datasets['val'] = val_ds
        if stage == 'test' or stage is None:
            test_ds = FontData(self.ds, size=self.size)
            self.datasets['test'] = test_ds 
        if stage == 'predict' or stage is None:
            predict_ds = FontData(self.ds, size=self.size)
            self.datasets['predict'] = predict_ds
        return 

    def train_dataloader(self):
        return DataLoader(self.datasets['train'], batch_size=self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.datasets['val'], batch_size=self.batch_size, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.datasets['test'], batch_size=self.batch_size, drop_last=False)

    def predict_dataloader(self):
        return DataLoader(self.datasets['predict'], batch_size=self.batch_size, drop_last=False)
