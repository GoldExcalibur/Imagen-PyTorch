import os, sys, argparse
from os.path import dirname, abspath, realpath, join
base_dir = dirname(dirname(realpath(__file__)))
sys.path.insert(0, base_dir)
from omegaconf import OmegaConf

import pyarrow 
import pyarrow.parquet as pq 
import io
from PIL import Image 
import random 

from ldm.data.harry_reader import Text2ImageDataLoader
from ldm.util import instantiate_from_config
from utils.hdfs_utils import hdfs_ls, hdfs_put 
from utils.utils import im2byte, mkdirs

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

def demo_data(pq_pth):
    dset = fetch_hdfs_cached(pq_pth)
    dtable = pq.read_table(dset)

    df = dtable.slice(0, 100).to_pandas()
    df = df.iloc[:20:5]

    vis_dir = join(base_dir, 'vis', 'tuchong5e')
    mkdirs(vis_dir)

    for idx, row in df.iterrows():
        import pdb; pdb.set_trace()
        image_id = row['image_id']
        im = Image.open(io.BytesIO(row['imbytes'])).convert('RGB')
        text = row['keywords'] 
        # text = row['remark']
        im.save(join(vis_dir, '{}.png'.format(text[:16])))
        # imbyte = row['imbytes']

def bytes2im(imbytes):
    buffer = io.BytesIO(imbytes)
    im_pil = Image.open(buffer).convert('RGB')
    return im_pil

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
        pil_col = df[self.image_key].apply(bytes2im)
        tensor_list = pil_col.apply(self.transform).to_list()
        im_tensor = torch.stack(tensor_list, dim=0)
        return {'image': im_tensor}

class Text2Tensor(IDF2Tensor):
    def __init__(self, text_key, tokenizer='bert-base-uncased'):
        self.text_key = text_key

        if not isinstance(tokenizer, str):
            self.tokernizer = tokenizer 
        else:
            # self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            self.tokenizer = T5Tokenizer.from_pretrained(tokenizer)
        # self.output_fields = ['input_ids', 'token_type_ids', 'attention_mask']
        self.output_fields = ['input_ids', 'attention_mask']
    
    def __call__(self, df):
        text_list = df[self.text_key].to_list()
        return {'text': text_list}

        # tok_dict = self.tokenizer(
        #     text_list, 
        #     padding='max_length',
        #     max_length=50, truncation=True
        # )
        # tok_keys = list(tok_dict.keys())
        # tensor_dict = {k: torch.LongTensor(v) for k,v in tok_dict.items()}
        # return tensor_dict

class MergePreprocess(IDF2Tensor):
    def __init__(self, df2tensor1, df2tensor2):
        self.output_fields = df2tensor1.output_fields + df2tensor2.output_fields 
        self.fn1 = df2tensor1 
        self.fn2 = df2tensor2 
        print(self.fn1.output_fields, self.fn2.output_fields)

    def __call__(self, df):
        tdict1 = self.fn1(df)
        tdict2 = self.fn2(df)
        keys1 = list(tdict1.keys())
        keys2 = list(tdict2.keys())
        print(keys1, keys2)
        return {**tdict1, **tdict2}
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser for test data')
    parser.add_argument('--base', type=str, required=True, help='base config')
    opt, unknown = parser.parse_known_args()
    cli = OmegaConf.from_dotlist(unknown)

    config = OmegaConf.load(opt.base)
    config = OmegaConf.merge(config, cli)
    print(config.data)
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()

    train_loader = data.train_dataloader()
    print(len(train_loader))
    batch_len_list = []
    for idx, batch in enumerate(train_loader):
        if idx > 1000: break
        # for k,v in batch.items():
            # print(k, type(v))
        # print(len(batch['c_crossattn']), len(batch['image']))
        batch_len_list.append(len(batch['c_crossattn']))
    print(min(batch_len_list), max(batch_len_list),)
    import pdb; pdb.set_trace()
    assert 0

    hdfs_src = 'hdfs://haruna/home/byte_arnold_lq_vc/yinweidong/zmldata/20220101/tuchong500m-image'
    # hdfs_src = "hdfs://haruna/home/byte_arnold_lq_vc/zhouboyan/datasets/multimodal/coco/parquets/valid"
    pq_list = hdfs_ls(hdfs_src)
    pq_list = [pth for pth in pq_list if pth.endswith('.parquet')]
    print(len(pq_list))

    # demo_data(pq_list[0])
    rand_pq_pth = random.choice(pq_list)
    # data_src = fetch_hdfs_cached(rand_pq_pth)
    # df = pq.read_table(data_src).to_pandas()
    # df = df.slice(0, 100).to_pandas()
    # print(df.shape)

    tokenizer_pth = '/mnt/bd/yinzihaodata/pretrained_models/mt5_xl'
    df2tensor = MergePreprocess(
        Image2Tensor('imbytes', size=64), 
        Text2Tensor('keywords', tokenizer_pth)
    )
    # tdict = df2tensor(df)

    tconfig = TConfig(rand_pq_pth)
    # batch_size = array_batch_size * array_size
    tloader = TableLoader(shuffle=True, drop_last=True, \
        num_worker=8, array_batch_size=8, array_size=1)
    # dloader = tloader.load_tsingle(tconfig, df2tensor, 0)

    sconfig = StreamConfig(stream_name='tuchong_val', num_epoch=5, is_mmap=True)
    tstream = LongRunTStream(sconfig, tconfig, 0)
    dloader = tloader.load_tstream(tstream, df2tensor)
    dloader = FakeEpochLoader(dloader, epoch_len=1000)

    # for tdict in dloader:
        # import pdb; pdb.set_trace()

    # diter = iter(dloader)
    # tdict = next(diter)
    # # diter.close()
    # print({k: v.size() for k,v in tdict.items()})








