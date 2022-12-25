from dataloader import KVReader
from PIL import Image 
from torch.utils.data import Dataset, Subset
from torchvision import transforms
import numpy as np 
import random 

class ArnoldDataReader(Dataset):
    def __init__(self, path, size=None, flip_p=0.5) -> None:
        super().__init__()
        self.num_parallel_reader = 2
        self.reader = KVReader(path, self.num_parallel_reader)
        self.keys = self.reader.list_keys()
        
    
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        index = [self.keys[idx]]
        data = self.reader.read_many(index)

        example = dict()
        example['file_path_'] = self.keys[idx]
        image = Image.open(data[0])
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=Image.BICUBIC)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example



class ArnoldSuperResolutionReader(Dataset):
    def __init__(self, path, size=None, target_size=64) -> None:
        self.path = path 

        super().__init__()
        self.num_parallel_reader = 2
        self.reader = KVReader(path, self.num_parallel_reader)
        self.keys = self.reader.list_keys()

    def __len__(self):
        return len(self.keys())

    
    def __getitem__(self, idx):
        index = [self.keys[idx]]
        data = self.reader.read_many(index)
        image = Image.open(data[0])
        example = dict()
        example['file_path_'] = self.keys[idx]

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
            example["c_concat"] = example["c_concat"] + np.random.normal(0, 0.04, example["c_concat"].shape)
        return example



