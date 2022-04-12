import os
import cv2
import torch
import numpy as np
from skimage import segmentation

from seg_kanezaki.code.utils import common as common_utils
from seg_kanezaki.code.utils.data_preprocessing import process_img


def SLIC_segmetation(im, compactness_value, num_superpixels):
	# Superpixels labels
	labels = segmentation.slic(im, compactness=compactness_value, n_segments=num_superpixels)
	# Superpixels boundaries
	boundaries = segmentation.mark_boundaries(im, labels)
	# Reshape labels into 1D
	labels = labels.reshape(im.shape[0] * im.shape[1])

	return labels, boundaries


def FZ_segmentation(im, sigma, scale, min_size):
	labels = segmentation.felzenszwalb(im, sigma=sigma, scale=scale, min_size=min_size)
	boundaries = segmentation.mark_boundaries(im, labels)
	labels = labels.reshape(im.shape[0] * im.shape[1])
	return labels, boundaries


def train_superpixel(args):

	# Clear previous session
	torch.cuda.empty_cache()

	# LOAD IMAGE
	im = process_img(args)
	data = torch.from_numpy(np.array([im.transpose((2, 0, 1)).astype('float32') / 255.]))
	data = torch.autograd.Variable(data)
	data = data.cuda()
	
	# If minlabels not specified, load number of classes from label
	if not args.COMMON.MINLABELS :
		args.COMMON.MINLABELS = len(np.unique(cv2.imread(args.LABEL_PATH, 0)))
	
	# APPLY SUPERPIXEL SEGMENTATION
	if args.COMMON.SUPERPIXEL_ALGO == "SLIC":
		labels, boundaries = SLIC_segmetation(im, args.COMMON.COMPACTNESS_VALUE, args.COMMON.NUM_SUPERPIXELS)
	elif args.COMMON.SUPERPIXEL_ALGO == "Felzenszwalb":
		labels, boundaries = FZ_segmentation(im, args.COMMON.FZ_SIGMA, args.COMMON.FZ_SCALE, args.COMMON.FZ_MIN_SIZE)

	# Extract unique labels
	u_labels = np.unique(labels)
	# l_inds CONSISTS OF N SUPERPIXELS AS NUMPY ARRAYS.EACH ELEMENT CONTAINS LOCATIONS OF EACH SUPERPIXEL.
	l_inds = []
	for i in range(len(u_labels)):
		l_inds.append(np.where(labels == u_labels[i])[0])
	
	# TRAIN MODEL
	# Initialize model
	model = common_utils.get_model(args, data.size(1))

	loss_fn = torch.nn.CrossEntropyLoss()
	optimizer = common_utils.get_optimizer(args, model)
	# Begin training
	label_colours = np.random.randint(255, size=(args.COMMON.NCHANNEL, 3))
	for batch_idx in range(args.COMMON.MAXITER):
		im = process_img(args)
		data = torch.from_numpy(np.array([im.transpose((2, 0, 1)).astype('float32') / 255.]))
		data = torch.autograd.Variable(data)
		data = data.cuda()
		# forwarding
		optimizer.zero_grad()
		output = model(data)[0]
		output = output.permute(1, 2, 0).contiguous().view(-1, args.COMMON.NCHANNEL)
		ignore, target = torch.max(output, 1)
		im_target = target.data.cpu().numpy()
		nLabels = len(np.unique(im_target))
		# Superpixel refinement
		for i in range(len(l_inds)):
			labels_per_sp = im_target[l_inds[i]]
			u_labels_per_sp = np.unique(labels_per_sp)
			hist = np.zeros(len(u_labels_per_sp))
			for j in range(len(hist)):
				hist[j] = len(np.where(labels_per_sp == u_labels_per_sp[j])[0])
			im_target[l_inds[i]] = u_labels_per_sp[np.argmax(hist)]
		target = torch.from_numpy(im_target)
		target = target.cuda()
		target = torch.autograd.Variable(target)
		loss = loss_fn(output, target)
		loss.backward()
		optimizer.step()
		print (batch_idx, '/', args.COMMON.MAXITER, ':', nLabels, loss.item())
		if nLabels <= args.COMMON.MINLABELS:
			print ("nLabels", nLabels, "reached minLabels", args.COMMON.MINLABELS, ".")
			break
			
	torch.save(model.state_dict(), os.path.join(args.DIR_SAVE_PATH, 'model_' + str(args.COMMON.MAXITER) + '.pth'))
	
	# Visualize
	# DESIGNATE RANDOM COLORS TO CLASSES
	output = model(data)[0]
	output = output.permute(1, 2, 0).contiguous().view(-1, args.COMMON.NCHANNEL)
	ignore, target = torch.max(output, 1)
	im_target = target.data.cpu().numpy()
	output_rgb = np.array([label_colours[c % 100] for c in im_target])
	output_rgb = output_rgb.reshape(im.shape)
	
	cv2.imwrite(args.PRED_SAVE_PATH, output_rgb)

	del model
