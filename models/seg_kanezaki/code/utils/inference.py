import os
import cv2
import tqdm
import torch
import numpy as np

from seg_kanezaki.code.utils import common as common_utils
from seg_kanezaki.code.utils.data_preprocessing import process_img


def get_iou_model(args, predictions, labels, n_classes, labels_name=None):
	"""
	Do matching between the values of the labels and the values of the output
	of the model and measure IOU metric for each matching.
	Save the mask of the prediction with the best matching applied.
	Args:
		args: Config dict
		predictions: Numpy array of predictions from the model
		labels: Numpy array with the labels
		n_classes: Int with number of classe of dataset
		labels_name: Optional list with filename info
	Return:
		IOU metrics
	"""
	# iou_model = None
	masks_list = []
	iou_imgs = []
	# Get IOU of each image
	for id_img in range(len(predictions)):
		iou, mask, _ = common_utils.get_iou(labels[id_img], predictions[id_img], n_classes, None)
		if labels_name:
			iou['img_name'] = os.path.basename(labels_name[id_img])
		iou_imgs.append(iou)
		masks_list.append(mask)
	iou_model = {
		'overall_mean': np.mean(np.array([x['mean_IOU'] for x in iou_imgs])),
		'variance': np.var(np.array([x['mean_IOU'] for x in iou_imgs])),
		'all': iou_imgs}
	# Save masks
	folder_path = os.path.join(args.DIR_SAVE_PATH, 'masks')
	if not os.path.exists(folder_path):
		print(folder_path)
		# Set up folder
		os.makedirs(folder_path)
	if labels_name:
		[cv2.imwrite(os.path.join(folder_path, "mask_" + os.path.basename(labels_name[id_x]).split('.')[0] + '.png'),
		             masks_list[id_x]) for id_x in range(len(masks_list))]
		[cv2.imwrite(os.path.join(folder_path, "label_" + os.path.basename(labels_name[id_x]).split('.')[0] + '.png'),
		             labels[id_x]) for id_x in range(len(labels))]
	return iou_model


def inference_folder(args, model, test_img_list, label_colours=None):
	"""
	Run inference over the given images
	Args:
		args: Config dict
		model: Torch model
		test_img_list: List of paths of the images to be predicted
		label_colours: Colour palette applied to the predictions for viasualization purposes
	"""
	print('Testing ' + str(len(test_img_list)) + ' images.')
	for img_file in tqdm.tqdm(test_img_list):
		# Load image
		args.IMG_PATH = img_file
		im = process_img(args, validation=True)
		data = torch.from_numpy(np.array([im.transpose((2, 0, 1)).astype('float32') / 255.]))
		data = data.cuda()
		data = torch.autograd.Variable(data)
		# Run inference
		output = model(data)[0]
		output = output.permute(1, 2, 0).contiguous().view(-1, args.COMMON.NCHANNEL)
		ignore, target = torch.max(output, 1)
		inds = target.data.cpu().numpy().reshape((im.shape[0], im.shape[1]))
		# Apply colours
		if label_colours is not None:
			inds = np.array([label_colours[c % args.COMMON.NCHANNEL] for c in inds])
			inds = inds.reshape(im.shape).astype(np.uint8)
			inds = cv2.cvtColor(inds, cv2.COLOR_BGR2GRAY)
		else:
			inds = inds.reshape((im.shape[0], im.shape[1])).astype(np.uint8)
		# Save mask
		pred_path = os.path.join(args.DIR_SAVE_PATH, os.path.basename(img_file).split('.')[0]+'.png')
		cv2.imwrite(pred_path, inds)
