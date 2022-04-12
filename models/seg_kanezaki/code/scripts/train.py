# from __future__ import print_function
import os
import cv2
import json
from glob import glob
import pytz
import optuna
import argparse
import datetime
from optuna.samplers import TPESampler
from easydict import EasyDict as edict

from seg_kanezaki.code.utils import superpixel_train as sp_training
from seg_kanezaki.code.utils import seg_tip_train as st_training
from seg_kanezaki.code.utils import common
from seg_kanezaki.code.utils import inference as inference_utils


def launch_superpixel_refinement(trial):
	"""
	Manage Superpixel refinement launch
	"""
	args = edict(**global_args)
	
	time_now = datetime.datetime.now(pytz.timezone('Europe/Madrid')).strftime('%Y-%m-%d_%H%M%S')
	experiment_folder = 'SP_{}_{}_{}'.format(args.DATA_INFO, args.TRAIN_CONFIGURATION, time_now)

	if trial:
		args.COMMON.DATA_AUGM = common.get_optuna_suggest_value(trial, "data_augm", args.OPTUNA_T.DATA_AUGM, args.OPTUNA_V.DATA_AUGM)
		args.COMMON.NCHANNEL = common.get_optuna_suggest_value(trial, "nchannel", args.OPTUNA_T.NCHANNEL, args.OPTUNA_V.NCHANNEL)
		args.COMMON.MAXITER = common.get_optuna_suggest_value(trial, "maxiter", args.OPTUNA_T.MAXITER, args.OPTUNA_V.MAXITER)
		args.COMMON.LR = common.get_optuna_suggest_value(trial, "lr", args.OPTUNA_T.LR, args.OPTUNA_V.LR)
		args.COMMON.NCONV = common.get_optuna_suggest_value(trial, "nconv", args.OPTUNA_T.NCONV, args.OPTUNA_V.NCONV)
		args.COMMON.NUM_SUPERPIXELS = common.get_optuna_suggest_value(trial, "num_superpixels", args.OPTUNA_T.NUM_SUPERPIXELS,
		                                                       args.OPTUNA_V.NUM_SUPERPIXELS)
		args.COMMON.COMPACTNESS_VALUE = common.get_optuna_suggest_value(trial, "compactness_value",
		                                                                args.OPTUNA_T.COMPACTNESS_VALUE,
		                                                                args.OPTUNA_V.COMPACTNESS_VALUE)
		print("OPTUNA")

	preds_list = []
	# Train just one image
	if args.TRAIN_CONFIGURATION == 'base':
		# Set up variables and folders
		args.IMG_PATH = args.BASE.IMG_PATH
		args.LABEL_PATH = args.BASE.LABEL_PATH[:-4]+'.png'
		args.DIR_SAVE_PATH = os.path.join(args.BASE.PRED_SAVE_PATH, experiment_folder, )
		os.mkdir(args.DIR_SAVE_PATH)
		with open(os.path.join(args.DIR_SAVE_PATH, 'train_params.json'), 'w') as fp:
			json.dump(args, fp, indent=4)
		args.PRED_SAVE_PATH = os.path.join(args.DIR_SAVE_PATH, os.path.basename(args.IMG_PATH)[:-4]) + '.png'

		# Run training
		sp_training.train_superpixel(args)
		
		# Get mean IOU
		predictions_list = [cv2.imread(args.PRED_SAVE_PATH)]
		labels_list = [cv2.imread(os.path.join(args.BASE.DIR_LABEL_PATH, os.path.basename(args.IMG_PATH)[:-4]+'.png'))]
		iou = inference_utils.get_iou_model(args, predictions_list, labels_list, args.COMMON.MINLABELS,
		                                         labels_name=[args.IMG_PATH])
		mean_iou = iou["overall_mean"]

	# Train a different model for each image
	elif args.TRAIN_CONFIGURATION == 'base_folder':
		# Set up variables and folders
		list_paths = sorted(os.listdir(args.BASE_FOLDER.DIR_PATH))
		args.DIR_SAVE_PATH = os.path.join(args.BASE_FOLDER.DIR_SAVE_PATH, experiment_folder)
		os.mkdir(args.DIR_SAVE_PATH)
		with open(os.path.join(args.DIR_SAVE_PATH, 'train_params.json'), 'w') as fp:
			json.dump(args, fp, indent=4)
		args.DIR_PATH = args.BASE_FOLDER.DIR_PATH
		# Train
		for i, img_path in enumerate(list_paths):
			print("Image {}/{}".format(i, len(list_paths)))
			args.IMG_PATH = os.path.join(args.BASE_FOLDER.DIR_PATH, img_path)
			args.LABEL_PATH = os.path.join(args.BASE_FOLDER.DIR_LABEL_PATH, os.path.basename(args.IMG_PATH)[:-4]) + '.png'
			args.PRED_SAVE_PATH = os.path.join(args.DIR_SAVE_PATH, os.path.basename(args.IMG_PATH)[:-4]) + '.png'
			# Run training
			sp_training.train_superpixel(args)
			preds_list.append(args.PRED_SAVE_PATH)
			
		# Get mean IOU
		labels_list = [cv2.imread(os.path.join(args.BASE_FOLDER.DIR_LABEL_PATH, x)) for x in sorted(os.listdir(args.BASE_FOLDER.DIR_LABEL_PATH))]
		predictions_list = [cv2.imread(x) for x in preds_list]
		iou = inference_utils.get_iou_model(args, predictions_list, labels_list, args.COMMON.MINLABELS,
		                                         labels_name=list_paths)
		mean_iou = iou["overall_mean"]
	else:
		raise ValueError("Not supported training configuration. Revise config_SP.py file")

	# Get mean IOU value for optuna optimization
	if trial:
		trial.report(mean_iou, 0)
		print("Final IOU: {}".format(mean_iou))
		return mean_iou
		
		
def launch_segmentation_tip(trial):
	"""
		Manage Segmentation tip launch
	"""
	args = edict(**global_args)
	
	time_now = datetime.datetime.now(pytz.timezone('Europe/Madrid')).strftime('%Y-%m-%d_%H%M%S')
	experiment_folder = 'ST_{}_{}_{}'.format(args.DATA_INFO, args.TRAIN_CONFIGURATION, time_now)
	
	if trial:
		args.COMMON.DATA_AUGM = common.get_optuna_suggest_value(trial, "data_augm", args.OPTUNA_T.DATA_AUGM, args.OPTUNA_V.DATA_AUGM)
		args.COMMON.NCHANNEL = common.get_optuna_suggest_value(trial, "nchannel", args.OPTUNA_T.NCHANNEL, args.OPTUNA_V.NCHANNEL)
		if args.TRAIN_CONFIGURATION != 'ref':
			args.COMMON.MAXITER = common.get_optuna_suggest_value(trial, "maxiter", args.OPTUNA_T.MAXITER, args.OPTUNA_V.MAXITER)
		else:
			args.REF.MAXUPDATE = common.get_optuna_suggest_value(trial, "maxiter", args.OPTUNA_T.MAXITER, args.OPTUNA_V.MAXITER)
			args.MAXITER = 1
			
		args.COMMON.LR = common.get_optuna_suggest_value(trial, "lr", args.OPTUNA_T.LR, args.OPTUNA_V.LR)
		args.COMMON.NCONV = common.get_optuna_suggest_value(trial, "nconv", args.OPTUNA_T.NCONV, args.OPTUNA_V.NCONV)
		args.COMMON.STEPSIZE_SIM = common.get_optuna_suggest_value(trial, "stepsize_sim", args.OPTUNA_T.STEPSIZE_SIM,
		                                                           args.OPTUNA_V.STEPSIZE_SIM)
		args.COMMON.STEPSIZE_CON = common.get_optuna_suggest_value(trial, "stepsize_con", args.OPTUNA_T.STEPSIZE_CON,
		                                                           args.OPTUNA_V.STEPSIZE_CON)
		args.COMMON.STEPSIZE_SCR = common.get_optuna_suggest_value(trial, "stepsize_scr", args.OPTUNA_T.STEPSIZE_SCR,
		                                                           args.OPTUNA_V.STEPSIZE_SCR)
		print("OPTUNA")
		
	preds_list = []
	# Train just one image
	if args.TRAIN_CONFIGURATION == 'base':
		# Set up variables and folders
		args.IMG_PATH = args.BASE.IMG_PATH
		args.LABEL_PATH = args.BASE.LABEL_PATH[:-4]+'.png'
		args.SCRIBBLE_PATH = args.BASE.SCRIBBLE_PATH
		args.DIR_SAVE_PATH = os.path.join(args.BASE.PRED_SAVE_PATH, experiment_folder, )
		os.mkdir(args.DIR_SAVE_PATH)
		with open(os.path.join(args.DIR_SAVE_PATH, 'train_params.json'), 'w') as fp:
			json.dump(args, fp, indent=4)
		args.PRED_SAVE_PATH = os.path.join(args.DIR_SAVE_PATH,'_'.join(os.path.basename(args.IMG_PATH).split('.')[:-1])) + '.png'
		# Train
		st_training.train_segtip(args)
		
		# Get mean IOU
		predictions_list = [cv2.imread(args.PRED_SAVE_PATH)]
		labels_list = [cv2.imread(os.path.join(args.BASE.DIR_LABEL_PATH, os.path.basename(args.IMG_PATH)))]
		iou = inference_utils.get_iou_model(args, predictions_list, labels_list, args.COMMON.MINLABELS,
		                                    labels_name=[args.IMG_PATH])
		mean_iou = iou["overall_mean"]
	
	# Train a different model for each image
	elif args.TRAIN_CONFIGURATION == 'base_folder':
		# Set up variables and folders
		args.SCRIBBLE_PATH = args.BASE_FOLDER.DIR_SCRIBBLE_PATH
		list_paths = sorted(os.listdir(args.BASE_FOLDER.DIR_PATH))
		args.DIR_SAVE_PATH = os.path.join(args.BASE_FOLDER.DIR_SAVE_PATH, experiment_folder)
		os.mkdir(args.DIR_SAVE_PATH)
		with open(os.path.join(args.DIR_SAVE_PATH, 'train_params.json'), 'w') as fp:
			json.dump(args, fp, indent=4)
		# Train
		for i, img_path in enumerate(list_paths):
			print("Image {}/{}".format(i, len(list_paths)))
			args.DIR_PATH = args.BASE_FOLDER.DIR_PATH
			args.IMG_PATH = os.path.join(args.BASE_FOLDER.DIR_PATH, img_path)
			args.LABEL_PATH = os.path.join(args.BASE_FOLDER.DIR_LABEL_PATH, img_path[:-4]+'.png')
			if args.SCRIBBLE_PATH != '':
				args.SCRIBBLE_PATH = os.path.join(args.BASE_FOLDER.DIR_SCRIBBLE_PATH, img_path)[:-4]+'.png'
			args.PRED_SAVE_PATH = os.path.join(args.DIR_SAVE_PATH,
											   '_'.join(os.path.basename(args.IMG_PATH).split('.')[:-1]) ) + '.png'
			# Run training for list_paths[i]
			st_training.train_segtip(args)
			preds_list.append(args.PRED_SAVE_PATH)
			
		# Get mean IOU
		
		predictions_list = [cv2.imread(x) for x in preds_list]
		labels_list = [cv2.imread(os.path.join(args.BASE_FOLDER.DIR_LABEL_PATH, x)) for x in sorted(os.listdir(args.BASE_FOLDER.DIR_LABEL_PATH))]
		iou = inference_utils.get_iou_model(args, predictions_list, labels_list, args.COMMON.MINLABELS,
		                                         labels_name=list_paths)
		mean_iou = iou["overall_mean"]
	
	# Train with reference
	elif args.TRAIN_CONFIGURATION == 'ref':
		# Set up variables and folders
		args.DIR_SAVE_PATH = os.path.join(args.REF.DIR_SAVE_PATH, experiment_folder, )
		os.mkdir(args.DIR_SAVE_PATH)
		with open(os.path.join(args.DIR_SAVE_PATH, 'train_params.json'), 'w') as fp:
			json.dump(args, fp, indent=4)
		# Run training for the reference images
		st_training.train_segtip_ref(args)
		# Get mean IOU
		predictions_list = [cv2.imread(x, 0) for x in sorted(glob(os.path.join(args.DIR_SAVE_PATH, "*.png")))]
		labels_list = [cv2.imread(os.path.join(args.REF.DIR_LABEL_PATH, x), 0) for x in sorted(os.listdir(args.REF.DIR_LABEL_PATH))]
		iou = inference_utils.get_iou_model(args, predictions_list, labels_list, args.COMMON.MINLABELS,
		                                         labels_name=sorted(os.listdir(args.REF.DIR_LABEL_PATH)))
		mean_iou = iou["overall_mean"]
	
	else:
		raise ValueError("Not supported training configuration. Revise config_ST.py file")

	# Get mean IOU value for optuna optimization
	if trial:
		trial.report(mean_iou, 0)
		print("Final IOU: {}".format(mean_iou))
		return mean_iou


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Seg_kanezaki training")
	parser.add_argument("--method", type=str, required=True)
	flags = parser.parse_args()
	
	# Load config
	if flags.method == 'sp':
		from seg_kanezaki.config_SP import cfg
	elif flags.method == 'st':
		from seg_kanezaki.config_ST import cfg
	else:
		raise ValueError("Not supported training method. Avalilable methods: sp (superpixel refinement) and st (seg_kanezaki)")
	config = edict(**cfg)
	global global_args
	global_args = edict(**config)
	
	if config.OPTUNA:  # If optuna enabled: update values
		# Create study
		study = optuna.create_study(study_name=datetime.datetime.now(pytz.timezone('Europe/Madrid')).strftime('%Y-%m-%d_%H%M%S'),
		                            storage=None,
		                            direction="maximize",
		                            sampler=TPESampler())
		# launch training
		
		if flags.method == 'sp':
			study.optimize(launch_superpixel_refinement, n_trials=config.OPTUNA_V.N_TRIALS)
		elif flags.method == 'st':
			study.optimize(launch_segmentation_tip, n_trials=config.OPTUNA_V.N_TRIALS)
		# Show statistics
		pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
		complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
		print("Study statistics: ")
		print("  Number of finished trials: ", len(study.trials))
		print("  Number of pruned trials: ", len(pruned_trials))
		print("  Number of complete trials: ", len(complete_trials))
		print("Best trial:")
		trial = study.best_trial
		print("  Value: ", trial.value)
		print("  Params: ")
		for key, value in trial.params.items():
			print("    {}: {}".format(key, value))
	
	else:  # If Optuna disabled: Run just a single training
		if flags.method == 'sp':
			launch_superpixel_refinement(None)
		elif flags.method == 'st':
			launch_segmentation_tip(None)
