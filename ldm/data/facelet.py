import os, yaml, pickle, shutil, glob 
from PIL import Image 
from tqdm import tqdm 
from torch.utils.data import Dataset, Subset
from torchvision import transforms
import numpy as np 
import random
import torch
import pickle


class FaceBase(Dataset):
    def __init__(self, path, size=None, flip_p=0.5):
        super().__init__()
        self.path = path 
        self.files = glob.glob('{}/*.png'.format(self.path))
        self.size = size
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        image = Image.open(self.files[idx])
        example = dict()
        example['file_path_'] = self.files[idx]
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=Image.BICUBIC)

        #image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example

class FaceBasePickle(Dataset):
    def __init__(self, path, size=None, flip_p=0.5):
        super().__init__()
        self.path = path 
        self.files = glob.glob('{}/*.pkl'.format(self.path))
        self.size = size 
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        #pickle.loads()
        return


class FaceSuperResolution(Dataset):
    def __init__(self, path, size=None, target_size=64, std=0.03):
        super().__init__()
        self.path = path 
        self.files = glob.glob('{}/*.png'.format(self.path))
        self.size = size 
        self.target_size = target_size
        self.std = std
        #self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return len(self.files)
    

    def __getitem__(self, idx):
        image = Image.open(self.files[idx])
        example = dict()
        example['file_path_'] = self.files[idx]
        if self.size is not None:
            image_lr = image.resize((self.size, self.size), resample=Image.BICUBIC)
            image_lr = image_lr.resize((self.target_size, self.target_size), resample=Image.BICUBIC)

        
        image_hr = image.resize((self.target_size, self.target_size), resample=Image.BICUBIC)

        p = random.random()
        if p > 0.5:
            image_lr = transforms.functional.hflip(image_lr)
            image_hr = transforms.functional.hflip(image_hr)


        #example['image'] = image_hr 
        #example['c_concat'] = image_lr
        image_lr = np.array(image_lr).astype(np.float32)
        image_hr = np.array(image_hr).astype(np.float32)

        example["image"] = (image_hr / 127.5 - 1.0).astype(np.float32)
        example["c_concat"] =  (image_lr / 127.5 - 1.0).astype(np.float32)

        p = random.random()
        if p > 0.5:
            example["c_concat"] = example["c_concat"] + np.random.normal(0, self.std, example["c_concat"].shape)
        
        return example





    