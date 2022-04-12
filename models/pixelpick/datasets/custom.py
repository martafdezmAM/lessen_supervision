from glob import glob
import os
import torch
import cv2
import imageio
import numpy as np
from random import random
from abc import ABC, abstractmethod

from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomGrayscale


class _BaseCustomDataset(ABC, Dataset):

	def __init__(self, args, val=False, query=False):
		self.root_dir = args.dir_dataset
		self.val = val
		self.query = query
		self.dir_checkpoints = args.dir_checkpoints
		self.ignore_index = args.ignore_index
		self.crop_bar = args.crop_bar
		self.n_pixels_by_us = args.n_pixels_by_us

		# seg variables for data augmentation
		self.use_data_augm = args.use_augmented_dataset
		self.geometric_augmentations = args.augmentations["geometric"]
		self.photometric_augmentations = args.augmentations["photometric"]

		# load images and masks and measure the amount of pixels per image
		self.images, self.masks, self.n_pixels_total = self._load_paths()

		# If query path does not exist, create a new one with args.n_pixels_by_us random pixels
		if args.n_pixels_by_us != 0:
			if not os.path.exists(args.query_path):
				a = np.zeros(self.size_shape[0] * self.size_shape[1])
				p = np.random.permutation(self.size_shape[0] * self.size_shape[1])
				a[p[:args.n_pixels_by_us]] = 1
				a = a.reshape(self.size_shape[1], self.size_shape[0])
				self.queries = np.repeat(a[np.newaxis, :, :], len(self.images), axis=0)
				# Convert array to bool
				self.queries = np.array(self.queries, dtype=bool)
				np.save(args.query_path, self.queries, )
			else:
				self.queries = np.load(args.query_path)

	@abstractmethod
	def _load_paths(self):
		pass

	@abstractmethod
	def _load_files(self, idx):
		pass

	def __len__(self):
		# Return the length of the dataset
		return len(self.images)

	def __getitem__(self, idx):
		# Return the observation based on an index.
		# Ex. dataset[0] will return the first element from the dataset, in this case the image and the label.
		if torch.is_tensor(idx):
			idx = idx.tolist()

		image, mask = self._load_files(idx)
		image *= 255.0/image.max()  # Normalize between 0 and 1
		image = torch.from_numpy(image)
		mask = torch.from_numpy(mask)

		if self.val or self.n_pixels_by_us == 0:
			return {'x': image, 'y': mask}

		# else:
		query = self.queries[idx]
		query = torch.from_numpy(query.astype(np.float32))

		# if val or query dataset, do NOT do data augmentation
		if not any([self.val, self.query]) and self.use_data_augm:
			r = random()
			if r < 1 / 3:  # Geometric augmentations
				# get queries
				query = Image.fromarray(self.queries[idx].astype(np.uint8) * 255) if self.queries is not None else None
				# data augmentation
				image, mask, query = self._geometric_augmentations(image, mask, queries=query)

			elif r < 2 / 3:  # Photometric augmentations
				image = self._photometric_augmentations(image)
				image = torch.from_numpy(image)
			# else: no data augmentation applied

		return {'x': image, 'y': mask, 'queries': query}

	def _geometric_augmentations(self, x, y, queries=None):

		if self.geometric_augmentations["random_hflip"]:
			if random() > 0.5:
				x, y = TF.hflip(x), TF.hflip(y)
				if queries is not None:
					queries = TF.hflip(queries)

		if queries is not None:
			queries = torch.from_numpy(np.asarray(queries, dtype=np.uint8) // 255)
		else:
			queries = torch.tensor(0)

		return x, y, queries

	def _photometric_augmentations(self, x):

		if self.photometric_augmentations["random_grayscale"]:
			x = RandomGrayscale(0.2)(x)

		if self.photometric_augmentations["random_gaussian_blur"]:
			_, w, h = x.shape
			x = GaussianBlur(kernel_size=int((0.1 * min(w, h) // 2 * 2) + 1))(x)
		return x

	def label_queries(self, queries, nth_query=None):
		assert len(queries) == len(self.queries), f"{queries.shape}, {self.queries.shape}"
		previous = self.queries.sum()
		self.queries = np.logical_or(self.queries, queries)
		self.n_pixels_total = new = self.queries.sum()
		print(f"# labelled pixels is changed from {previous} to {new} (delta: {new - previous})")

		if isinstance(nth_query, int):
			# os.makedirs(f"{self.dir_checkpoints}/{nth_query}_query", exist_ok=True)
			np.save(f"{self.dir_checkpoints}/{nth_query}_query/label.npy", self.queries)


class GaussianBlur(object):
	# Implements Gaussian blur as described in the SimCLR paper
	def __init__(self, kernel_size, min=0.1, max=2.0):
		self.min = min
		self.max = max
		# kernel size is set to be 10% of the image height/width
		self.kernel_size = kernel_size

	def __call__(self, sample):
		sample = np.array(sample)

		# blur the image with a 50% chance
		prob = np.random.random_sample()

		if prob < 0.5:
			sigma = (self.max - self.min) * np.random.random_sample() + self.min
			sample = cv2.GaussianBlur(sample.transpose(1, 2, 0), (self.kernel_size, self.kernel_size), sigma).transpose(2, 0, 1)

		return sample

from PIL import Image
class MetalDAMDataset(_BaseCustomDataset):
	def __init__(self, args, val=False, test=False, query=False):
		self.splits = args.splits
		self.test = test
		self.size_shape = (1024, 768)
		self.n_classes = args.n_classes
		#self.size_shape = (480, 384,)
		super(MetalDAMDataset, self).__init__(args, val=val, query=query)

	def _load_paths(self):
		if self.splits:
			if self.val:
				with open(os.path.join(self.root_dir, 'val.txt')) as f:
					list_paths = f.read().splitlines()
			elif self.test:
				with open(os.path.join(self.root_dir, 'test.txt')) as f:
					list_paths = f.read().splitlines()
			else:  # Train
				with open(os.path.join(self.root_dir, 'train.txt')) as f:
					list_paths = f.read().splitlines()

			images = sorted([os.path.join(self.root_dir, 'images/', lbl_name.split('.')[0] + ".jpg")
							 for lbl_name in list_paths])
			masks = sorted([os.path.join(self.root_dir, 'labels/', lbl_name.split('.')[0] + ".png")
							for lbl_name in list_paths])
		else:
			images = sorted(glob(os.path.join(self.root_dir, 'images/*.jpg')))
			masks = sorted(glob(os.path.join(self.root_dir, 'labels/*.png')))
		shape_img = cv2.imread(images[0])[:-self.crop_bar].shape
		n_pixels_total = shape_img[0] * shape_img[1]

		assert len(images) == len(masks) and len(masks) > 0

		return images, masks, n_pixels_total

	def _load_files(self, idx):
		img_name = self.images[idx]
		mask_name = self.masks[idx]
		# Load image and mask
		image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)[:-self.crop_bar]  # Load and crop image bar
		mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)

		# Resize
		image = np.array(Image.fromarray(image, mode="L").resize(self.size_shape, Image.BILINEAR))
		image = image.astype(np.float32)
		img_rgb = np.zeros((3, *image.shape, ), image.dtype)
		img_rgb[2, :, :] = img_rgb[1, :, :] = img_rgb[0, :, :] = image
		mask = mask.astype(np.uint8)
		mask = Image.fromarray(mask, mode="L")
		mask = mask.resize(self.size_shape, Image.NEAREST)
		mask = np.array(mask, dtype=np.int64)
		
		return np.array(img_rgb), np.array(mask)


class FloodnetDataset(_BaseCustomDataset):
	def __init__(self, args, val=False, test=False, query=False):
		self.crop_bar = 0
		self.splits = args.splits
		self.test = test
		self.size_shape = (768, 1024, )  # orig_size/4
		self.n_classes = args.n_classes
		super(FloodnetDataset, self).__init__(args, val=val, query=query)

	def _load_paths(self):
		if self.splits:
			if self.val or self.test:
				images = sorted(glob(os.path.join(self.root_dir, "test", "*.jpg")))
				masks = sorted(glob(os.path.join(self.root_dir, "testannot", "*.png")))
			else:  # Train
				images = sorted(glob(os.path.join(self.root_dir, "train", "*.jpg")))
				masks = sorted(glob(os.path.join(self.root_dir, "trainannot", "*.png")))
		else:
			images = sorted(glob(os.path.join(self.root_dir, 'train/*.jpg')))
			masks = sorted(glob(os.path.join(self.root_dir, 'mask/*.png')))
		shape_img = cv2.imread(images[0])[:-self.crop_bar].shape
		n_pixels_total = shape_img[0] * shape_img[1]

		assert len(images) == len(masks) and len(masks) > 0

		return images, masks, n_pixels_total

	def _load_files(self, idx):
		img_name = self.images[idx]
		mask_name = self.masks[idx]
		mask = cv2.imread(mask_name, 0)
		image = cv2.imread(img_name)[:mask.shape[0], :mask.shape[1], :]  # Load and crop image bar
		
		image = np.array(Image.fromarray(image, mode='RGB').resize(self.size_shape, Image.BILINEAR))
		image = image.astype(np.float32).transpose(2, 0, 1)
		mask = mask.astype(np.uint8)
		mask = Image.fromarray(mask, mode="L")
		mask = mask.resize(self.size_shape, Image.NEAREST)
		mask = np.array(mask, dtype=np.int64)

		return image, mask
