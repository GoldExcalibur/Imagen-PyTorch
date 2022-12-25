 

import ray


from torch.utils.data import DataLoader, dataloader, random_split
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from torch.utils.data import IterableDataset
import pytorch_lightning as pl
import io
import numpy as np 
import torch
import random
import cv2 
import pandas as pd 

from utils.hdfs_utils import hdfs_ls
from ldm.util import instantiate_from_config
from ldm.modules.image_degradation.bsrgan import add_blur, add_Gaussian_noise
from utils.utils import parse_tags

# change to cubic as in SR3
def process(image, size):
    if size is not None:
        image = image.resize((size, size), resample=Image.Resampling.BICUBIC)
    image = np.array(image).astype(np.uint8)
    image = (image / 127.5 -1.0).astype(np.float32)
    return image

def process_sr(image, size, target_size, blur_sf=None):
    '''
    size: low res; target_size: high res 
    '''
    if size is not None:
        image = image.resize((size, size), resample=Image.Resampling.BICUBIC)
    if blur_sf is not None:
        # print(f'gaussian blur {blur_sf:d} enabled !')
        image = add_blur(image, sf=blur_sf)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

    image = image.resize((target_size, target_size), resample=Image.Resampling.BICUBIC)
    image = np.array(image).astype(np.uint8)
    image = (image / 127.5 -1.0).astype(np.float32)
    # cv2 resize: might exceed [-1,1]
    # image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    return image 


## reader parquet files from path
class RayDataset(IterableDataset):
    def __init__(self, files, size=64, flip_p =0.5, shuffle=False, im_key='_c0'):
        super().__init__()
        self.files = files
        self.file = None
        self.size = size 
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.shuffle = shuffle
        self.im_key = im_key
    
    def __iter__(self):
        if self.file is not None:
            print('read parquet using ray: {}'.format(self.file))
            self.ds = ray.data.read_parquet(self.file)
            if self.shuffle is True:
                self.ds.random_shuffle()
            print('self.ds has counter: {}'.format(self.ds.count()))
        else:
            print('{} is none, and random select a file from {} files'.format(self.file, len(self.files)))
            random_id = random.randint(0, len(self.files)-1)
            self.file = self.files[random_id]
            self.ds = ray.data.read_parquet(self.file)
            if self.shuffle is True:
                self.ds.random_shuffle()
            print('self.ds has counter: {}'.format(self.ds.count()))

        for data in self.ds.iter_rows(): 
            image = Image.open(io.BytesIO(data[self.im_key]))
            image = self.flip(image)
            yield process(image, self.size)
    
    def get_data(self):
        for data in self.ds.iter_rows():
            image = Image.open(io.BytesIO(data[self.im_key]))
            yield image 

    def get_batch_data(self):
        for data in self.ds.iter_batches(batch_size=16):
            yield data

## error

def collect_fn():

    return

def worker_init_fn(_):
    print('initializing worker')
    worker_info = torch.utils.data.get_worker_info()
    files = worker_info.dataset.files
    print('total size of dataset: {}'.format(len(files)))
    random_id = random.randint(0, len(files)-1)
    worker_info.dataset.file = files[random_id]

    print('initializing dataset file: {}'.format(files[random_id]))

    


class RayReader(pl.LightningDataModule):
    def __init__(self, batch_size, hdfs_root, shuffle=False, num_workers=None, size=64, flip_p=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.hdfs_root = hdfs_root

        self.shuffle = shuffle
        self.size = size
        self.flip_p = flip_p
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        self.train_ds = None 
        self.val_ds = None 
        self.test_ds = None
        
    def prepare_data(self):
        self.files = hdfs_ls(self.hdfs_root)
        print('processing files: {}'.format(len(self.files)))
        return 
    

    def setup(self, stage=None):

        train_subset, test_subset, val_subset = random_split(self.files, [len(self.files)-2,1,1])
        train_files = [self.files[i] for i in train_subset.indices]
        test_files = [self.files[i] for i in test_subset.indices]
        val_files = [self.files[i] for i in val_subset.indices]

        if stage == 'fit' or stage is None:
            self.train_ds = RayDataset(train_files, size=self.size, flip_p = self.flip_p, shuffle=self.shuffle)
        if stage == 'test' or stage is None:
            self.test_ds = RayDataset(test_files, size=self.size, flip_p = self.flip_p, shuffle=self.shuffle)
        if stage == 'predict' or stage is None:
            self.predict_ds = RayDataset(val_files, size=self.size, flip_p=self.flip_p, shuffle=self.shuffle)
        return 
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(self.test_ds, batch_size = self.batch_size, drop_last=True)
    
    def test_dataloader(self):
        return DataLoader(self.predict_ds, batch_size = self.batch_size, drop_last=True)
    


class RayData(IterableDataset):
    def __init__(self, ds, size=64, flip_p=0.5, shuffle=False, im_key='_c0'):
        super().__init__()
        self.size = size 
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.shuffle=shuffle
        self.ds = ds
        self.im_key = im_key
    
    def __iter__(self):
        if self.shuffle is True:
            self.ds.random_shuffle()
        for data in self.ds.iter_rows():
            image = Image.open(io.BytesIO(data[self.im_key]))
            example = dict()
            example['image'] = process(image, self.size)
            yield example
    
    def __len__(self):
        return self.ds.count()


class RayDataLoader(pl.LightningDataModule):
    def __init__(self, batch_size, train_hdfs_root, test_hdfs_root=None, val_hdfs_root=None, shuffle=False, num_workers=None, size=64, flip_p=0.5, im_key='_c0'):
        super().__init__()
        self.batch_size = batch_size 
        self.train_hdfs_root = train_hdfs_root
        self.test_hdfs_root = test_hdfs_root
        self.val_hdfs_root = val_hdfs_root
        self.shuffle = shuffle 
        self.size = size 
        self.flip_p = flip_p
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        self.train_ds = None 
        self.val_ds = None 
        self.test_ds = None
        self.im_key = im_key 

    def prepare_data(self):
        

        return 
    
    def setup(self, stage=None):
        print('preparing setting up data: {}'.format(self.train_hdfs_root))        
        self.train_set = ray.data.read_parquet(self.train_hdfs_root)
        print('we get {} training data'.format(self.train_set.count()))
        if self.test_hdfs_root is not None:
            self.test_set = ray.data.read_parquet(self.test_hdfs_root)
            print('we get {} test data'.format(self.test_set.count()))
        if self.val_hdfs_root is not None:
            self.val_set = ray.data.read_parquet(self.val_hdfs_root)
            print('we get {} val data'.format(self.val_set.count()))
        self.datasets = dict()
        if stage=='fit' or stage is None:
            self.train_ds = RayData(self.train_set, size=self.size, flip_p=self.flip_p, shuffle=self.shuffle, im_key=self.im_key)
            self.datasets['train'] = self.train_ds
        if stage=='test' or stage is None:
            self.test_ds = RayData(self.test_set, size=self.size, flip_p=self.flip_p, shuffle=self.shuffle, im_key=self.im_key)
            self.datasets['test'] = self.test_ds
        if stage=='predict' or stage is None:
            self.predict_ds = RayData(self.val_set, size=self.size, flip_p=self.flip_p, shuffle=self.shuffle, im_key=self.im_key)
            self.datasets['predict'] = self.predict_ds

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, drop_last=True)#, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.test_ds, batch_size = self.batch_size, drop_last=True)#, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.predict_ds, batch_size=self.batch_size, drop_last=True)#, num_workers=self.num_workers)


class SRData(IterableDataset):
    def __init__(self, ds, size=32, flip_p=0.5, shuffle=False, target_size=128, im_key='_c0', blur=None) -> None:
        super().__init__()
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.shuffle = shuffle
        self.ds = ds 
        self.size = size
        self.target_size = target_size
        assert self.size < self.target_size, \
            'size of low-res im {:d} should be smaller than high-res {:d}'.format(self.size, self.target_size)
        self.flip_p = flip_p
        self.im_key = im_key
        self.blur = blur 
        
    
    def __iter__(self):
        if self.shuffle is True:
            self.ds.random_shuffle()
        
        for data in self.ds.iter_rows():
            image = data[self.im_key]
            if not isinstance(image, bytes): image = image.as_py()
            image = Image.open(io.BytesIO( image ))
            example = dict()
            
            example['image'] = process(image, self.target_size)
            example['c_concat'] = process_sr(image, self.size, self.target_size, blur_sf=self.blur)
            yield example
    
    def __len__(self):
        return self.ds.count()

    
class SRDataLoader(pl.LightningDataModule):
    def __init__(self, batch_size, train_hdfs_root, test_hdfs_root=None, val_hdfs_root=None, \
        shuffle=False, num_workers=None, size=32, target_size=128, flip_p=0.5, im_key='_c0', filter=False, blur=None):
        super().__init__()
        self.batch_size = batch_size 
        self.train_hdfs_root = train_hdfs_root
        self.test_hdfs_root = test_hdfs_root 
        self.val_hdfs_root = val_hdfs_root
        self.shuffle = shuffle
        self.size = size 
        self.flip_p = flip_p
        self.num_workers = num_workers if num_workers is not None else batch_size * 4
        self.train_ds = None 
        self.val_ds = None 
        self.test_ds = None 
        self.target_size = target_size
        self.im_key = im_key 
        self.filter = filter 
        self.blur = None if (blur is None or blur == 'None') else int(blur) 
        # print('blur in this dataloader {} !'.format(blur))
    
    def prepare_data(self):
        return 
    
    def size_filter_fn(self, batch):
        outs = []
        for x in batch[self.im_key]:
            if not isinstance(x, bytes): x = x.as_py()
            im = Image.open(io.BytesIO( x ))
            h, w = im.size 
            if h > self.target_size and w > self.target_size:
                outs.append(x)
        return outs 

    def get_hdfs_data(self, hdfs_root):
        ds = ray.data.read_parquet(hdfs_root)
        if self.filter:
            ds = ds.map_batches(self.size_filter_fn, batch_size=4096)
        return ds 

    def setup(self, stage = None):
        print('preparing setting up data: {}'.format(self.train_hdfs_root))        
        self.train_set = self.get_hdfs_data(self.train_hdfs_root)
        print('we get {} training data'.format(self.train_set.count()))
        
        if self.test_hdfs_root is None or self.test_hdfs_root == self.train_hdfs_root:
            self.test_set = self.train_set 
        else:
            self.test_set = self.get_hdfs_data(self.test_hdfs_root)
        print('we get {} test data'.format(self.test_set.count()))

        if self.val_hdfs_root is None or self.val_hdfs_root == self.train_hdfs_root:
            self.val_set = self.train_set 
        else:
            self.val_set = self.get_hdfs_data(self.val_hdfs_root)
        print('we get {} val data'.format(self.val_set.count()))
        
        self.datasets = dict()
        if stage=='fit' or stage is None:
            self.train_ds = SRData(self.train_set, size=self.size, flip_p=self.flip_p, shuffle=self.shuffle, target_size=self.target_size, im_key=self.im_key, blur=self.blur)
            self.datasets['train'] = self.train_ds
        if stage=='test' or stage is None:
            self.test_ds = SRData(self.test_set, size=self.size, flip_p=self.flip_p, shuffle=self.shuffle, target_size=self.target_size, im_key=self.im_key)
            self.datasets['test'] = self.test_ds
        
        if stage=='' or stage is None:
            self.eval_ds = None
        
        if stage=='predict' or stage is None:
            self.predict_ds = SRData(self.val_set, size=self.size, flip_p=self.flip_p, shuffle=self.shuffle, target_size=self.target_size, im_key=self.im_key)
            self.datasets['predict'] = self.predict_ds
        
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size = self.batch_size, drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, drop_last=True)
    
    def test_dataloader(self):
        return DataLoader(self.predict_ds, batch_size=self.batch_size, drop_last=True)
    
class Text2ImageData(IterableDataset):
    def __init__(self, ds, size=32, flip_p=0.5, shuffle=False, im_key='imbytes', text_key='keywords') -> None:
        super().__init__()
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.shuffle = shuffle
        self.ds = ds 
        self.size = size
        self.flip_p = flip_p
        self.im_key = im_key
        self.text_key = text_key
         
    
    def __iter__(self):
        if self.shuffle is True:
            self.ds.random_shuffle()
        
        for data in self.ds.iter_rows():
            # get image 
            image = data[self.im_key]
            if not isinstance(image, bytes): image = image.as_py()
            image = Image.open(io.BytesIO( image ))

            example = dict()
            example['image'] = process(image, self.size)
            example['c_crossattn'] = data[self.text_key]
            yield example
    
    def __len__(self):
        return self.ds.count()

class Text2ImageDataLoader(pl.LightningDataModule):
    def __init__(self, batch_size, train_hdfs_root, test_hdfs_root=None, val_hdfs_root=None,\
        shuffle=False, num_workers=None, size=64, im_key='imbytes', text_key='keywords', flip_p=0.5, filter=False):
        super().__init__()
        self.batch_size = batch_size
        self.train_hdfs_root = train_hdfs_root
        self.test_hdfs_root = test_hdfs_root 
        self.val_hdfs_root = val_hdfs_root 
        self.shuffle = shuffle 
        self.size = size 
        self.num_workers = num_workers if num_workers is not None else batch_size * 2 
        self.train_ds = None 
        self.val_ds = None 
        self.test_ds = None 
        self.im_key = im_key 
        self.text_key = text_key
        self.filter = filter 
        self.flip_p = flip_p


    def prepare_data(self):
        pass 

    def text_filter_fn(self, batch):
        outs = []
        if not isinstance(batch, pd.DataFrame):
            batch = batch.to_pandas()
        for idx, row in batch.iterrows():
            chn_tags, _ = parse_tags(row['keywords'])
            if len(chn_tags) == 0: continue 
            remark = row['remark']
            if len(remark) == 0: continue 
            outs.append(row)
        return outs 

    def get_hdfs_data(self, hdfs_root):
        ds = ray.data.read_parquet(hdfs_root)
        if self.filter:
            ds = ds.map_batches(self.text_filter_fn, batch_size=self.batch_size)
        return ds 

    def setup(self, stage = None):
        print('preparing setting up data: {}'.format(self.train_hdfs_root))        
        self.train_set = self.get_hdfs_data(self.train_hdfs_root)
        print('we get {} training data'.format(self.train_set.count()))

        if self.test_hdfs_root is None or self.test_hdfs_root == self.train_hdfs_root:
            self.test_set = self.train_set 
        else:
            self.test_set = self.get_hdfs_data(self.test_hdfs_root)
        print('we get {} test data'.format(self.test_set.count()))

        if self.val_hdfs_root is None or self.val_hdfs_root == self.train_hdfs_root:
            self.val_set = self.train_set 
        else:
            self.val_set = self.get_hdfs_data(self.val_hdfs_root)
        print('we get {} val data'.format(self.val_set.count()))
        
        self.datasets = dict()
        if stage=='fit' or stage is None:
            self.train_ds = Text2ImageData(self.train_set, size=self.size, flip_p=self.flip_p, shuffle=self.shuffle, im_key=self.im_key, text_key=self.text_key)
            self.datasets['train'] = self.train_ds
        if stage=='test' or stage is None:
            self.test_ds = Text2ImageData(self.test_set, size=self.size, flip_p=self.flip_p, shuffle=self.shuffle, im_key=self.im_key, text_key=self.text_key)
            self.datasets['test'] = self.test_ds
        
        if stage=='' or stage is None:
            self.eval_ds = None
        
        if stage=='predict' or stage is None:
            self.predict_ds = Text2ImageData(self.val_set, size=self.size, flip_p=self.flip_p, shuffle=self.shuffle, im_key=self.im_key, text_key=self.text_key)
            self.datasets['predict'] = self.predict_ds
        
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size = self.batch_size, drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, drop_last=True)
    
    def test_dataloader(self):
        return DataLoader(self.predict_ds, batch_size=self.batch_size, drop_last=True)


