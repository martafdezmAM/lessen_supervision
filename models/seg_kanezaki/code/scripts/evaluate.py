import os
import cv2
import json
import glob
import argparse
import numpy as np
from easydict import EasyDict as edict

from seg_kanezaki.code.utils import inference as inference_utils


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Seg_kanezaki inference")
	parser.add_argument("--imgs_path", type=str, required=True)
	parser.add_argument("--labels_path", type=str, required=True)
	parser.add_argument("--model_dir", type=str, required=True)
	flags = parser.parse_args()

	# Load config parameters
	train_params_file = os.path.join(flags.model_dir, 'train_params.json')
	with open(train_params_file) as f:
		config = json.load(f)
	config = edict(config)
	conf = edict(**config)
	conf.MODEL_DIR = flags.model_dir
	conf.TEST_DATA_DIR = flags.imgs_path
	conf.TEST_LABELS_DIR = flags.labels_path

	# Load ckpt
	ckpts = sorted(glob.glob(os.path.join(conf.MODEL_DIR, '*.pth')))  # Get all checkpoints
	
	# Load images and labels
	x_test = sorted(glob.glob(os.path.join(conf.TEST_DATA_DIR, '*')))
	
	conf.CKPT = ckpts[len(ckpts)-1]

	# Load the predictions
	img_names = sorted(glob.glob(os.path.join(conf.MODEL_DIR, '*.png')))
	predictions_paths = [os.path.join(conf.MODEL_DIR, os.path.basename(x)) for x in img_names]
	predictions_list = [cv2.imread(x, 0) for x in predictions_paths]
	if conf.DATA_INFO == 'floodnet':
		labels_paths = [os.path.join(conf.TEST_LABELS_DIR, os.path.basename(x).replace(".png", "_lab.png")) for x in img_names]
		labels_list = [np.array(cv2.imread(x, 0), np.uint8) for x in labels_paths]
		labels_list = [cv2.resize(im, (int(im.shape[1] / 2), int(im.shape[0] / 2))) for im in labels_list]
	else:
		labels_paths = [os.path.join(conf.TEST_LABELS_DIR, os.path.basename(x)) for x in img_names]
		labels_list = [np.array(cv2.imread(x, 0), np.uint8) for x in labels_paths]

	# Get IOU and save masks
	conf.DIR_SAVE_PATH = conf.MODEL_DIR
	iou = inference_utils.get_iou_model(conf, predictions_list, labels_list, conf.COMMON.MINLABELS, labels_name=x_test)
	# Save metrics
	with open(os.path.join(conf.MODEL_DIR, 'inference_iou.json'), 'w') as json_file:
		json.dump(iou, json_file, sort_keys=True, indent=4)
	mean_iou = iou['overall_mean']
	print("Final mean IOU: {}".format(mean_iou))
