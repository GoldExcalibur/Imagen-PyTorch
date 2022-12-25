import os, sys, argparse
from os.path import realpath, abspath, dirname, join
this_dir = dirname(realpath(__file__))
base_dir = dirname(this_dir)
sys.path.insert(0, base_dir)

import random 
from glob import glob 
import torch 
import torchvision as tv
from PIL import Image 
import torchvision.transforms as transforms
from utils.utils import custom_to_np

if __name__ == '__main__':
    random.seed(2022)
    
    log_dir = join(base_dir, 'logs')
    # sample_dir = join(log_dir, 'samples', '00382500', 'img', 'sr')
    # sample_dir = join(log_dir, '2022-07-02T12-46-38_background_256_ray_data/checkpoints/samples', \
    #     '00870001/2022-07-20-14-05-55', 'img')
    sample_dir = join(log_dir, '2022-07-25T12-56-56_font_80/checkpoints/samples', \
        '00160001/2022-07-27-11-34-41', 'img')

    # sample_dir = join(log_dir, 'samples', 'tuchong_background_2022-07-12-01-13/img')

    img_pths = []

    # ref_dir = join(log_dir, 'ilvr_reference/font')
    # ref_dir = join(log_dir, 'ilvr_reference/faces_real')
    # img_pths += sorted(glob(join(ref_dir, '*.png')))

    # sample_dir = join(log_dir, '2022-07-25T12-56-56_font_80/checkpoints/samples', '00080001/2022-07-26-13-27-03/img')
    # sample_dir = join(log_dir, '2022-07-25T12-56-56_font_80/checkpoints/samples', '00160001/2022-07-27-10-31-39/img')
    # sample_dir = join(log_dir, '2022-06-01T10-37-40_face_aligned_ray_data_baseline/checkpoints/samples', '00572617/2022-07-27-10-45-51/img')
    # sample_dir = join(log_dir, '2022-06-01T10-37-40_face_aligned_ray_data_baseline/checkpoints/samples', '00572617/2022-07-27-10-55-21/img')
    img_pths += sorted(glob(join(sample_dir, '*.png')))
    
    print(len(img_pths))
    # img_pths = random.sample(img_pths, 128)
    print(len(img_pths))

    channel_mode = 'RGB'
    # channel_mode = 'BGR'
    im_size = 80
    # im_size = 256
    imgs = [Image.open(p).convert(channel_mode).resize((im_size, im_size)) for p in img_pths]

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.ToTensor(), normalize
    ])

    # import pdb; pdb.set_trace()
    img_tensors = [transform(im) for im in imgs]
    img_grid = tv.utils.make_grid(img_tensors, nrow=16, padding=2, normalize=True, scale_each=True)
    # save_pth = join(base_dir, 'face_hr_256_{}.png'.format(channel_mode))
    # save_pth = join(base_dir, 'background_256_{}.png'.format(channel_mode))
    save_pth = join(base_dir, 'sample_font_80.png')
    # save_pth = join(base_dir, 'ilvr_sample_font_stride=8.png')
    # save_pth = join(base_dir, 'ilvr_sample_face_stride=4.png')
    tv.utils.save_image(img_grid, save_pth)

