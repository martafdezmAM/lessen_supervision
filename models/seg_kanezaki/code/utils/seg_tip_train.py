import os
import cv2
import datetime
import tqdm
import torch
import numpy as np

from seg_kanezaki.code.utils import common as common_utils
from seg_kanezaki.code.utils.inference import inference_folder
from seg_kanezaki.code.utils.data_preprocessing import process_img, process_scribble


def train_segtip(args):

	# Clear previous session
	torch.cuda.empty_cache()

	# LOAD IMAGE
	im = process_img(args)
	data = torch.from_numpy(np.array([im.transpose((2, 0, 1)).astype('float32') / 255.]))
	data = data.cuda()
	data = torch.autograd.Variable(data)
	
	# If minlabels not specified, load number of classes from label
	if not args.COMMON.MINLABELS :
		args.COMMON.MINLABELS = len(np.unique(cv2.imread(args.LABEL_PATH, 0)))
	
	# LOAD SCRIBBLE
	if args.SCRIBBLE_PATH != '':
		mask_inds, inds_sim, inds_scr, inds_sim, target_scr = process_scribble(args)
		print(np.unique(target_scr))
		inds_sim = inds_sim.cuda()
		inds_scr = inds_scr.cuda()
		target_scr = target_scr.cuda()
		target_scr = torch.autograd.Variable(target_scr)
		# set minLabels
		args.COMMON.MINLABELS = len(mask_inds)
		
	# TRAIN MODEL
	model = common_utils.get_model(args, data.size(1))

	# similarity loss definition
	loss_fn = torch.nn.CrossEntropyLoss()
	# scribble loss definition
	loss_fn_scr = torch.nn.CrossEntropyLoss()
	# continuity loss definition
	loss_hpy = torch.nn.L1Loss(size_average=True)
	loss_hpz = torch.nn.L1Loss(size_average=True)
	HPy_target = torch.zeros(im.shape[0] - 1, im.shape[1], args.COMMON.NCHANNEL)
	HPz_target = torch.zeros(im.shape[0], im.shape[1] - 1, args.COMMON.NCHANNEL)
	HPy_target = HPy_target.cuda()
	HPz_target = HPz_target.cuda()
	optimizer = common_utils.get_optimizer(args, model)
	# Begin training
	label_colours = np.random.randint(255, size=(args.COMMON.NCHANNEL, 3))
	for batch_idx in range(args.COMMON.MAXITER):
		# Load image
		im = process_img(args)
		data = torch.from_numpy(np.array([im.transpose((2, 0, 1)).astype('float32') / 255.]))
		data = data.cuda()
		data = torch.autograd.Variable(data)
		# forwarding
		optimizer.zero_grad()
		output = model(data)[0]
		output = output.permute(1, 2, 0).contiguous().view(-1, args.COMMON.NCHANNEL)
		outputHP = output.reshape((im.shape[0], im.shape[1], args.COMMON.NCHANNEL))
		HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
		HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
		lhpy = loss_hpy(HPy, HPy_target)
		lhpz = loss_hpz(HPz, HPz_target)
		ignore, target = torch.max(output, 1)
		im_target = target.data.cpu().numpy()
		nLabels = len(np.unique(im_target))
		
		# loss
		if args.SCRIBBLE_PATH != '':
			loss = args.COMMON.STEPSIZE_SIM * loss_fn(output[inds_sim], target[inds_sim]) + args.COMMON.STEPSIZE_SCR * loss_fn_scr(
				output[inds_scr], target_scr[inds_scr]) + args.COMMON.STEPSIZE_CON * (lhpy + lhpz)
		else:
			loss = args.COMMON.STEPSIZE_SIM * loss_fn(output, target) + args.COMMON.STEPSIZE_CON * (lhpy + lhpz)
		loss.backward()
		optimizer.step()
		if batch_idx % 50 == 0:
			print(batch_idx, '/', args.COMMON.MAXITER, '|', ' label num :', nLabels, ' | loss :', loss.item())
		if nLabels <= args.COMMON.MINLABELS:
			print("nLabels", nLabels, "reached minLabels", args.COMMON.MINLABELS, ".")
			break
			
	torch.save(model.state_dict(), os.path.join(args.DIR_SAVE_PATH, 'model_' + str(args.COMMON.MAXITER) + '.pth'))
	
	output = model(data)[0]
	output = output.permute(1, 2, 0).contiguous().view(-1, args.COMMON.NCHANNEL)
	ignore, target = torch.max(output, 1)
	im_target = target.data.cpu().numpy()
	im_target_rgb = np.array([label_colours[c % args.COMMON.NCHANNEL] for c in im_target])
	im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)
	
	cv2.imwrite(args.PRED_SAVE_PATH, im_target_rgb)

	del model


def train_segtip_ref(args):
	# Clear previous session
	torch.cuda.empty_cache()

	# LOAD DATA
	img_list = os.listdir(args.REF.TRAIN_DIR_PATH)
	img_list = [os.path.join(args.REF.TRAIN_DIR_PATH, x) for x in img_list]
	# Get first image
	args.IMG_PATH = img_list[0]
	im = process_img(args)
	# TRAIN MODEL
	model = common_utils.get_model(args, im.shape[2])
	del im

	# similarity loss definition
	loss_fn = torch.nn.CrossEntropyLoss()
	# scribble loss definition
	loss_fn_scr = torch.nn.CrossEntropyLoss()
	# continuity loss definition
	loss_hpy = torch.nn.L1Loss(size_average=True)
	loss_hpz = torch.nn.L1Loss(size_average=True)
	optimizer = common_utils.get_optimizer(args, model)
	# Begin training
	for batch_idx in range(args.COMMON.MAXITER):
		print('Training started:' + str(datetime.datetime.now()) + ' ' + str(batch_idx + 1) + ' / ' + str(args.COMMON.MAXITER))
		for im_file in range(int(len(img_list) / args.REF.BATCH_SIZE)):
			for loop in tqdm.tqdm(range(args.REF.MAXUPDATE)):
				im = []
				for batch_count in range(args.REF.BATCH_SIZE):
					# Get first image
					args.IMG_PATH = img_list[args.REF.BATCH_SIZE * im_file + batch_count]
					processed_im = process_img(args)
					processed_im = processed_im.transpose((2, 0, 1)).astype('float32') / 255.
					im.append(processed_im)
					# LOAD SCRIBBLE
					if args.REF.DIR_SCRIBBLE_PATH != '':
						args.SCRIBBLE_PATH = os.path.join(args.REF.DIR_SCRIBBLE_PATH,
						                                  os.path.basename(img_list[args.REF.BATCH_SIZE *
						                                                            im_file + batch_count])[:-4]+'.png')
						mask_inds, inds_sim, inds_scr, inds_sim, target_scr = process_scribble(args)
						inds_sim = inds_sim.cuda()
						inds_scr = inds_scr.cuda()
						target_scr = target_scr.cuda()
						target_scr = torch.autograd.Variable(target_scr)
						# set minLabels
						args.COMMON.MINLABELS = len(mask_inds)
						
				data = torch.from_numpy(np.array(im))
				data = data.cuda()
				data = torch.autograd.Variable(data)
				HPy_target = torch.zeros(processed_im.shape[1] - 1, processed_im.shape[2], args.COMMON.NCHANNEL)
				HPz_target = torch.zeros(processed_im.shape[1], processed_im.shape[2] - 1, args.COMMON.NCHANNEL)
				HPy_target = HPy_target.cuda()
				HPz_target = HPz_target.cuda()
				# forwarding
				optimizer.zero_grad()
				output = model(data)[0]
				output = output.permute(1, 2, 0).contiguous().view(-1, args.COMMON.NCHANNEL)
				outputHP = output.reshape((processed_im.shape[1], processed_im.shape[2], args.COMMON.NCHANNEL))
				HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
				HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
				lhpy = loss_hpy(HPy, HPy_target)
				lhpz = loss_hpz(HPz, HPz_target)
				ignore, target = torch.max(output, 1)
				# Get total loss
				if args.REF.DIR_SCRIBBLE_PATH !=  '':
					loss = args.COMMON.STEPSIZE_SIM * loss_fn(output[inds_sim], target[inds_sim]) + \
					       args.COMMON.STEPSIZE_SCR * loss_fn_scr(output[inds_scr], target_scr[inds_scr]) + \
					       args.COMMON.STEPSIZE_CON * (lhpy + lhpz)
				else:
					loss = args.COMMON.STEPSIZE_SIM * loss_fn(output, target) + args.COMMON.STEPSIZE_CON * (lhpy + lhpz)
				loss.backward()
				optimizer.step()
			print(loss.data.cpu().numpy())
			im_target = target.data.cpu().numpy()
			nLabels = len(np.unique(im_target))
			print("nLabels: ", nLabels,)
			
			if nLabels <= args.COMMON.MINLABELS:
				print("nLabels", nLabels, "reached minLabels", args.COMMON.MINLABELS, ".")
				break
			
		torch.save(model.state_dict(), os.path.join(args.DIR_SAVE_PATH, 'model_' + str(batch_idx) + '.pth'))
		
		inference_folder(args, model, img_list)

	del HPy, HPz, lhpy, lhpz, HPy_target, HPz_target, data, model
