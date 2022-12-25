import argparse, sys, os
from os.path import join, isdir, isfile, exists 

import numpy as np 
import torch 
from PIL import Image
import io
import skimage
import random
from matplotlib import pyplot as plt
import face_alignment
import copy
import cv2
import time 
import pyarrow as pa
import pyarrow.parquet as pq 
from tqdm import tqdm
from collections import defaultdict, OrderedDict
import torch
import torchvision.utils as vutils
import torchvision.transforms as trans

import ray
os.environ['RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE'] = '1'
MAX_MEM = 1024 * 1024 * 1024 * 32
ray.init(ignore_reinit_error=True, _memory=MAX_MEM, _driver_object_store_memory=MAX_MEM, object_store_memory=MAX_MEM, _redis_max_memory=MAX_MEM)

from utils.hdfs_utils import hdfs_mkdirs, hdfs_put, hdfs_ls
from utils.utils import get_fname, save_vis, get_hdfs_subset
from utils.utils import plot_box, plot_landmarks
from utils.utils import im2byte, DB_Saver

base_dir='/mnt/bd/yinzihaodata'
code_dir=join(base_dir, 'code')
data_dir=join(base_dir, 'data')
pretrained_dir=join(base_dir, 'pretrained_models')

def process(image, size, dtype=np.float32):
	assert dtype in [np.float32, np.uint8]
	if size is not None: # bicubic as used in sr3
		image = image.resize((size, size), resample=Image.Resampling.BICUBIC)
		# image = trans.functional.resize(
		# 	image, (size, size), \
		# 	interpolation=trans.InterpolationMode.BICUBIC, \
		# 	antialias=True
		# )

	image = np.array(image).astype(np.uint8) #0-255
	if dtype == np.float32:
		image = (image / 127.5 -1.0).astype(np.float32)
	return image

#For dlib’s 68-point facial landmark detector:
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("inner_mouth", (60, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])

#For dlib’s 5-point facial landmark detector:
FACIAL_LANDMARKS_5_IDXS = OrderedDict([
	("right_eye", (2, 3)),
	("left_eye", (0, 1)),
	("nose", (4))
])

class box_FaceAligner(object):
	def __init__(self, predictor, im_size=(128,128), tgt_box=[28,48,100,115]):
		self.predictor = predictor 
		self.im_w, self.im_h = im_size 
		assert self.im_w == 128 and self.im_h == 128 
		self.tgt_box = tgt_box 

	@staticmethod
	def get_box(pt, bounds):
		minx, miny, maxx, maxy = bounds
		tlx, tly = pt.min(axis=0)
		brx, bry = pt.max(axis=0)
		tlx = min(max(tlx, minx), maxx)
		tly = min(max(tly, miny), maxy)
		brx = min(max(brx, minx), maxx)
		bry = min(max(bry, miny), maxy)
		return [int(tlx), int(tly), int(brx), int(bry)]

	@staticmethod 
	def box2affine(box, tgt_box):
		tlx, tly, brx, bry = box
		src_pts = [[tlx, tly], [tlx, bry], [brx, bry]]
		src_pts = np.array([np.array(pt) for pt in src_pts])
		
		tlx, tly, brx, bry = tgt_box
		dst_pts = [[tlx, tly], [tlx, bry], [brx, bry]]
		dst_pts = np.array([np.array(pt) for pt in dst_pts])
		return cv2.getAffineTransform(src_pts.astype(np.float32), dst_pts.astype(np.float32))

	def align(self, image):
		# image: npy (h, w, ch)
		h, w, ch = image.shape 
		pt = self.predictor.get_landmarks(image) 
		if pt is None: return None 
		else: pt = pt[0]

		box = self.get_box(pt, [0, 0, w, h])
		M = self.box2affine(box, self.tgt_box)
		output = cv2.warpAffine(image, M, (self.im_h, self.im_w)) 
		return {'im_M': output, 'landmark': pt, 'box': box}
		
class eye_FaceAligner(object):
	def __init__(self, predictor, desiredLeftEye=(0.35, 0.35), desiredFaceWidth=256, desiredFaceHeight=None):
		# store the facial landmark predictor, desired output left
		# eye position, and desired output face width + height
		self.predictor = predictor
		self.desiredLeftEye = desiredLeftEye
		self.desiredFaceWidth = desiredFaceWidth
		self.desiredFaceHeight = desiredFaceHeight
		self.eye_width_ratio = 0.15

		# if the desired face height is None, set it to be the
		# desired face width (normal behavior)
		if self.desiredFaceHeight is None:
			self.desiredFaceHeight = self.desiredFaceWidth
	
	def align(self, image):
		# convert the landmark (x, y)-coordinates to a NumPy array
		h, w, ch = image.shape
		pt = self.predictor.get_landmarks(image)
		if pt is None: return None
		
		pt = pt[0]
		#simple hack ;)
		if len(pt)==68:
			# extract the left and right eye (x, y)-coordinates
			(lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
			(rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
		else:
			(lStart, lEnd) = FACIAL_LANDMARKS_5_IDXS["left_eye"]
			(rStart, rEnd) = FACIAL_LANDMARKS_5_IDXS["right_eye"]
			
		leftEyePts = pt[lStart:lEnd]
		rightEyePts = pt[rStart:rEnd]

		# compute the center of mass for each eye
		leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
		rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
		# compute the angle between the eye centroids
		dY = rightEyeCenter[1] - leftEyeCenter[1]
		dX = rightEyeCenter[0] - leftEyeCenter[0]

		dist = np.sqrt((dX ** 2) + (dY ** 2))
		if dist < min(h,w) * self.eye_width_ratio:
			return None
		angle = np.degrees(np.arctan2(dY, dX)) - 180

		# compute the desired right eye x-coordinate based on the
		# desired x-coordinate of the left eye
		desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

		# determine the scale of the new resulting image by taking
		# the ratio of the distance between eyes in the *current*
		# image to the ratio of distance between eyes in the
		# *desired* image
		desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
		desiredDist *= self.desiredFaceWidth
		scale = desiredDist / dist

		# compute center (x, y)-coordinates (i.e., the median point)
		# between the two eyes in the input image
		eyesCenter = ( int((leftEyeCenter[0] + rightEyeCenter[0]) // 2),
			int((leftEyeCenter[1] + rightEyeCenter[1]) // 2) )
		# print(eyesCenter, angle, scale)

		# grab the rotation matrix for rotating and scaling the face
		M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

		# update the translation component of the matrix
		tX = self.desiredFaceWidth * 0.5
		tY = self.desiredFaceHeight * self.desiredLeftEye[1]
		M[0, 2] += (tX - eyesCenter[0])
		M[1, 2] += (tY - eyesCenter[1])

		# apply the affine transformation
		(w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
		output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

		# return the aligned face
		return {'im_M': output, 'landmark': pt}

def get_affine_transform(center,
						 scale,
						 rot,
						 output_size,
						 shift=np.array([0, 0], dtype=np.float32),
						 inv=0, pixel_std=200.0):
	if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
		print(scale)
		scale = np.array([scale, scale])

	scale_tmp = scale * pixel_std
	src_w = scale_tmp[0]
	dst_w = output_size[0]
	dst_h = output_size[1]

	rot_rad = np.pi * rot / 180
	src_dir = get_dir([0, src_w * -0.5], rot_rad)
	dst_dir = np.array([0, dst_w * -0.5], np.float32)

	src = np.zeros((3, 2), dtype=np.float32)
	dst = np.zeros((3, 2), dtype=np.float32)
	src[0, :] = center + scale_tmp * shift
	src[1, :] = center + src_dir + scale_tmp * shift
	dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
	dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

	src[2:, :] = get_3rd_point(src[0, :], src[1, :])
	dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

	if inv:
		trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
	else:
		trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

	return trans

def prepare_face_net(detector_type='sfd'):
	ori_dir = torch.hub.get_dir()
	torch.hub.set_dir(pretrained_dir)
	print(f'set torch cache dir from {ori_dir:s} -> {pretrained_dir:s}')
	pre_cp_pth = join(pretrained_dir, 'checkpoints', 's3fd-619a316812.pth') 

	device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
	net = face_alignment.FaceAlignment(
		face_alignment.LandmarksType._2D,  
		flip_input=False, face_detector=detector_type, device=device,
		face_detector_kwargs={
			'path_to_detector': pre_cp_pth,
			'filter_threshold' : 0.8,
		}
	)
	return net

def get_batch(data):
	ims = [Image.open(io.BytesIO(i)) for i in data] # pil.image 
	
	return ims 

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='argument parse for face alignment')
	parser.add_argument('--size', type=int, default=128, help='tgt size for image')
	parser.add_argument('--n_pts', type=int, default=68, help='number of keypoints')
	parser.add_argument('--align_method', type=str, choices=['box', 'eye'], help='align method type (face bbox or eye keypoints)')
	parser.add_argument('--save_name', type=str, required=True, help='specify hdfs root name')
	parser.add_argument('--src_hpth', type=str, required=True, help='src dir for parquet')
	parser.add_argument('--dst_hpth', type=str, \
		default='hdfs://haruna/home/byte_ailab_vc_video_summarization/user/yinzihao/data', 
		help='dst dir to save parquet'
	)
	parser.add_argument('--save_freq', type=int, default=10000, help='save freq to hdfs')
	parser.add_argument('--im_key', type=str, default='_c0', help='image key for src dset')
	args = parser.parse_args()

	# 6M face data processed
	# tuchong_hpth = "hdfs://haruna/home/byte_ailab_vc_video_summarization/common_dataset/tuchong/"
	# tuchong_hpth = "hdfs://haruna/home/byte_ailab_vc_video_summarization/user/yinzihao/data/tuchong/"
	# yzh_hpth = "hdfs://haruna/home/byte_ailab_vc_video_summarization/user/yinzihao"
	# save_dir = join(args.dst_hpth, 'data', 'tuchong', args.save_name)
	save_dir = join(args.dst_hpth, args.save_name)
	hdfs_mkdirs(save_dir)
	# cmd = f'hdfs dfs -mkdir {save_dir:s}'
	# os.system(cmd) 
	# cmd = f'hdfs dfs -ls {save_dir:s}'
	# os.system(cmd) 

	cw_dir = os.getcwd()
	print(f'current working dir: {cw_dir}')
	vis_dir = join(cw_dir, 'vis')
	if not exists(vis_dir): os.makedirs(vis_dir)

	# subset_list = get_hdfs_subset(args.src_hpth, suffix='.parquet')
	subset_list = hdfs_ls(args.src_hpth)
	subset_list = [p for p in subset_list if p.endswith('.parquet')]

	print('{} === HAS ==> {} parquets'.format(args.src_hpth, len(subset_list)))

	tgt_box = [int(pos * args.size // 128) for pos in [28, 48, 100, 115]] # this tgt box is obtained under 128*128

	net = prepare_face_net(detector_type='sfd')
	if args.align_method == 'box':
		kwargs = {'im_size':(args.size, args.size), 'tgt_box':[28, 48, 100, 115]}
	else: # default 0.35
		kwargs = {'desiredLeftEye':(0.37, 0.45), 'desiredFaceWidth': args.size, 'desiredFaceHeight': args.size}

	aligner = eval(f'{args.align_method}_FaceAligner')(net, **kwargs)

	db_saver = DB_Saver(['image'], args.save_freq, save_dir)
	for sidx, subset_pth in enumerate(subset_list):
		dset = ray.data.read_parquet(subset_pth)
		dset_len = dset.count()
		print('{} dset length: {}'.format(sidx, dset_len))
		vis_freq = max(1, int(dset_len/50.0))
		cur_vis_dir = join(vis_dir, get_fname(subset_pth), 'align_by_{}'.format(args.align_method))
		if not exists(cur_vis_dir): os.makedirs(cur_vis_dir)

		for idx, data in tqdm(enumerate(dset.iter_rows()), position=0, leave=True):
		# for idx, batch in tqdm(enumerate(dset.iter_batches, position=0, leave=True)):
			im = data[args.im_key]
			im = Image.open(io.BytesIO(im))
			h, w = im.size 
			if h <= args.size and w <= args.size:
				continue 
			# fix me: when return image in [0,1], no landmark detected
			im = process(im, args.size, dtype=np.uint8)
			out_dict = aligner.align(im)
			# skip when pt is None or some profile faces
			if out_dict is None: 
				continue 
			  
			pt = out_dict['landmark']
			im_M = out_dict['im_M']
			# print(im.shape, im_M.shape, pt.shape)
			db_saver.append(
				image= im2byte(im_M),
			)
			
			if idx % vis_freq == 0:
				im_pt = plot_landmarks(im, pt)
				vis_list = [im, im_pt, im_M]
				if 'box' in out_dict:
					im_box = plot_box(im, out_dict['box']) 
					vis_list.insert(2, im_box)
				save_pth = join(cur_vis_dir, '{:s}_{:d}_face.jpg'.format(
					args.align_method, idx))
				save_vis(vis_list, save_pth)

			if len(db_saver) >= args.save_freq:
				db_saver.save()

		# vutils.save_image(images,
			# join(cur_vis_dir, 'im_pre_align.jpg'), normalize=True, nrow=10, scale_each=True)
		# vutils.save_image(images_M,
			# join(cur_vis_dir, 'im_post_align.jpg'), normalize=True, nrow=10, scale_each=True)

	# ray.shutdown()


		

   
	


	