import cv2
import torch
import random
import numpy as np


def process_img(args, validation=False):
    
	im = cv2.imread(args.IMG_PATH)

	if args.COMMON.CROP_BAR_SZ != 0:
		im = im[:-args.COMMON.CROP_BAR_SZ]

	if args.COMMON.BLUR:
		im = cv2.medianBlur(im, 5)

	im = cv2.resize(im, (int(im.shape[1]/32)*32, int(im.shape[0]/32)*32))
	if args.DATA_INFO == 'floodnet':
		labe_shape = cv2.imread(args.IMG_PATH.replace("_images", "_masks").replace(".jpg", "_lab.png"), 0).shape
		im = cv2.resize(im[:labe_shape[0], :labe_shape[1], :], (1024, 768,))
		
	if args.COMMON.DATA_AUGM and not validation:
		lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
		lab_planes = cv2.split(lab)
		val_clip = random.randint(2, 9)
		val_grid = random.randint(3, 9)
		clahe = cv2.createCLAHE(clipLimit=val_clip, tileGridSize=(val_grid, val_grid))
		lab_planes[0] = clahe.apply(lab_planes[0])
		lab = cv2.merge(lab_planes)
		
		im = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
	elif args.COMMON.CLAHE:
		bgr = im.copy()
		lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
		lab_planes = cv2.split(lab)
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
		lab_planes[0] = clahe.apply(lab_planes[0])
		lab = cv2.merge(lab_planes)
		im = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

	return im


def process_scribble(args):
	mask = cv2.imread(args.SCRIBBLE_PATH, -1)
	if args.COMMON.CROP_BAR_SZ != 0:
		mask = mask[:-args.COMMON.CROP_BAR_SZ]

	if args.MODEL != "kanezaki":  # Resize to multiple of 32
		mask = cv2.resize(mask, (int(mask.shape[1]/32)*32, int(mask.shape[0]/32)*32))

	if args.DATA_INFO == 'floodnet':
		mask = cv2.resize(mask, (1024, 768,),  interpolation = cv2.INTER_NEAREST)

	mask = mask.reshape(-1)
	mask_inds = np.unique(mask)
	mask_inds = np.delete(mask_inds, np.argwhere(mask_inds == 255))
	inds_sim = torch.from_numpy(np.where(mask == 255)[0])
	inds_scr = torch.from_numpy(np.where(mask != 255)[0])
	target_scr = torch.from_numpy(mask.astype(np.int))

	return mask_inds, inds_sim, inds_scr, inds_sim, target_scr

