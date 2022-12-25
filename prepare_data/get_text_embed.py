import sys, os, argparse 
from os.path import join, isdir, isfile, dirname, exists, realpath 
base= dirname(dirname(realpath(__file__)))
sys.path.insert(0, base)

import io 
from PIL import Image 
import ray 
os.environ['RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE'] = '1'

import pandas as pd 
import torch
import transformers 
from transformers import T5Tokenizer, T5EncoderModel, T5Config, BertTokenizer 
from tqdm import tqdm 
from collections import defaultdict
import pyarrow as pa
import random 

from utils.hdfs_utils import hdfs_mkdirs, hdfs_put, hdfs_ls
from utils.utils import get_hdfs_subset, save_vis, get_fname, im2byte, process_image, mkdirs, is_chn, is_eng 
from utils.utils import tensor2byte, byte2tensor, parse_tags

# config 
MAX_LENGTH = 256 
DEFAULT_T5_NAME = 'google'
HTTP_PROXY='http://bj-rd-proxy.byted.org:3128'

# def get_tokenizer(name, cache_dir=None):
#     tokenizer = T5Tokenizer.from_pretrained(name, cache_dir=cache_dir)
#     return tokenizer 

# def get_model(name, cache_dir=None):
#     model = T5EncoderModel.from_pretrained(name, cache_dir=cache_dir)
#     return model 

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

def get_tokenizer(cp_dir, eval_mode=True):
    tokenizer = BertTokenizer.from_pretrained(cp_dir)
    # tokenizer = T5Tokenizer.from_pretrained(cp_dir)
    # if eval_mode: 
    #     tokenizer.eval()
    # else:
    #     tokenizer.train()

    # if torch.cuda.is_available():
    #     tokenizer = tokenizer.cuda()
    return tokenizer 

def get_model(cp_dir, eval_mode=True):
    encoder = T5EncoderModel.from_pretrained(cp_dir)
    if eval_mode: 
        encoder.eval()
    else: 
        encoder.train()

    if torch.cuda.is_available():
        encoder = encoder.cuda() 
    return encoder 

@torch.no_grad()
def t5_encode_text(opt, texts, models_dict, pad_type):
    device = next(models_dict['encoder'].parameters()).device

    encoded = models_dict['tokenizer'].batch_encode_plus(
        texts, return_tensors='pt', padding=pad_type, 
        max_length=opt.max_len, truncation=True,
    )

    input_ids = encoded.input_ids.to(device)
    attn_mask = encoded.attention_mask.to(device)

    output = models_dict['encoder'](input_ids = input_ids, attention_mask = attn_mask)
    encoded_text = output.last_hidden_state #.detach() # b,l,c
    return encoded_text, attn_mask.bool()

def toy_example(opt, models_dict):
    texts = ['messi is playing football', 'cristiano is goat', 'spanish is the champion']
    texts += ['a child screaming at finding a worm within a half-eaten apple',
        'lizard running across the desert on two feet',
        'waking up to a psychedelic landscape',
        'seashells sparkling in the shallow waters']
    print([len(t) for t in texts])
    for text in texts:
        text_embeds, text_masks = t5_encode_text(opt, [text], models_dict, opt.pad_type)
        print(len(text))
        print(text_embeds.size(), text_embeds.requires_grad)
        print(text_masks.size(), text_masks.requires_grad)
        print('#' * 80)
    # import pdb; pdb.set_trace()

def run_check(opt, models_dict, save_dir):
    # opt.src_hpth = 'hdfs://haruna/home/byte_ailab_vc_video_summarization/user/yinzihao/data/tuchong/tuchong_background'
    # opt.src_hpth = 'hdfs://haruna/home/byte_ailab_vc_video_summarization/user/yinzihao/data/tuchong/image_64_chn_keywords'
    subset_list = hdfs_ls(opt.src_hpth)
    subset_list = [p for p in subset_list if p.endswith('.parquet')]
    print('{} === HAS ==> {} parquets'.format(opt.src_hpth, len(subset_list)))

    ray.init(ignore_reinit_error=True)
    subset_pth = random.sample(subset_list, 1)[0]
    dset = ray.data.read_parquet(subset_pth)
    dset_len = dset.count()
    print(subset_pth, dset_len)
    
    for row in dset.iter_rows():
        im = Image.open(io.BytesIO(row['image']))
        batch_texts = row['text']
        batch_text_embeds = byte2tensor(row['text_embed'], [opt.max_len, -1])
        batch_attn_masks = byte2tensor(row['attn_mask'], [opt.max_len,])

        gt_text_embeds, gt_attn_masks = t5_encode_text(opt, [batch_texts], models_dict, opt.pad_type)
        import pdb; pdb.set_trace()


def run(opt, models_dict, save_dir):
    subset_list = hdfs_ls(opt.src_hpth)
    subset_list = [p for p in subset_list if p.endswith('.parquet')]
    print('{} === HAS ==> {} parquets'.format(opt.src_hpth, len(subset_list)))
    
    db_saver = DB_Saver(['image', 'text', 'text_embed', 'attn_mask'], opt.save_freq, save_dir)
    ray.init(ignore_reinit_error=True)
    for didx, subset_pth in enumerate(subset_list):
        dset = ray.data.read_parquet(subset_pth)
        dset_len = dset.count() 
        print('{} len of {} is {}'.format(didx, subset_pth, dset_len))

        for bidx, batch in tqdm(enumerate(dset.iter_batches(batch_size=opt.batch_size)), desc='iter on subset'):
            # here batch is pd dframe 
            batch_texts = []
            batch_imbytes = []
            
            if not isinstance(batch, pd.DataFrame): batch = batch.to_pandas()
            for ridx, row in batch.iterrows():
                chn_tags, _ = parse_tags(row['keywords'])
                text = ' '.join([str(c) for c in chn_tags])
                # text = row['remark']
                if len(text) == 0: continue 
                batch_texts.append( text )

                x = row['imbytes']
                if not isinstance(x, bytes): x = x.as_py()
                x = Image.open(io.BytesIO(x))
                x = x.resize((opt.im_size, opt.im_size))
                batch_imbytes.append( im2byte(x) )
                       
            if len(batch_texts) == 0: continue 

            batch_text_embeds, batch_attn_masks = t5_encode_text(opt, batch_texts, models_dict, opt.pad_type)
            batch_text_embeds = [tensor2byte(tb) for tb in batch_text_embeds]
            batch_attn_masks = [tensor2byte(tm) for tm in batch_attn_masks]

            # to check decoded bytes correctness 
            # decoded_text_embeds = [byte2tensor(tb, [opt.max_len, -1]) for tb in encoded_text_embeds]
            # decoded_attn_masks = [byte2tensor(tm, [opt.max_len]) for tm in encoded_attn_masks]


            # batch_imbytes = [batch_imbytes[idx] for idx in batch_idxs]
            db_saver.append(
                image=batch_imbytes, 
                text=batch_texts, #[bytes(t, 'utf-8') for t in batch_texts],
                text_embed=batch_text_embeds, 
                attn_mask=batch_attn_masks,
            )    

            if len(db_saver) >= opt.save_freq:
                db_saver.save()
                import pdb; pdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args parser for prepare text embedding')
    parser.add_argument('--max_len', default=256, type=int, help='max length of sentence embedding ?')
    parser.add_argument('--t5_name', default='', help='text model name')
    parser.add_argument('--cache_dir', default='/mnt/bd/yinzihaodata/pretrained_models',\
        type=str, help='dir for save pretrained model')
    parser.add_argument('--eval_mode', type=int, default=1, help='model set to eval')
    parser.add_argument('--pad_type', type=str, default='max_length', choices=['max_length', 'longest', 'do_not_pad'], help='pad type for batch_encode_plus')
    parser.add_argument('--src_hpth', type=str, required=True, help='src hdfs data set')
    parser.add_argument('--dst_hpth', type=str, \
        default='hdfs://haruna/home/byte_ailab_vc_video_summarization/user/yinzihao/data',
        help='dst dir to save parquet'
    )
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--save_name', type=str, required=True, help='should specify name relative to dst_hpth')
    parser.add_argument('--save_freq', type=int, default=65536, help='save frequency')
    parser.add_argument('--im_size', type=int, default=64, help='target image size')
    opt, unknown = parser.parse_known_args()

    # mkdir hdfs dir 
    save_dir = join(opt.dst_hpth, opt.save_name)
    hdfs_mkdirs(save_dir)

    # load text model 
    cp_dir = join(opt.cache_dir, opt.t5_name)

    # tokenizer = T5Tokenizer.from_pretrained(cp_dir)
    # t5 = T5EncoderModel.from_pretrained(cp_dir)
    print('eval mode: {} pad type: {}'.format(opt.eval_mode, opt.pad_type))
    models_dict = {}
    if opt.t5_name != '' and isdir(cp_dir):
        models_dict['tokenizer'] = get_tokenizer(cp_dir, opt.eval_mode)
    else:
        print(f'{cp_dir} for tokenizer not exists !')

    if opt.t5_name != '' and isdir(cp_dir):
        models_dict['encoder'] = get_model(cp_dir, opt.eval_mode)
    else:
        print(f'{cp_dir} for encoder not exists')
    
    # toy example 
    # toy_example(opt, models_dict)

    # run(opt, models_dict, save_dir)
    run_check(opt, models_dict, save_dir)







