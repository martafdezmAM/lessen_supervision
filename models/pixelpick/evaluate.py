import os
import json
import cv2
import numpy as np
import pickle as pkl
from tqdm import tqdm
from copy import deepcopy
from math import ceil
from argparse import ArgumentParser, Namespace

import torch
import torch.nn.functional as F

from utils.utils import Visualiser, get_dataloader
from utils.metrics import EvaluationMetrics, RunningScore
from query import UncertaintySampler


@torch.no_grad()
def evaluate(args_train, args_infer, dataloader, model):
	metrics = EvaluationMetrics(args_train.n_classes)
	running_score = RunningScore(args_train.n_classes)
	vis = Visualiser(args_train.dataset_name)
	list_results = []
	dataloader_iter, tbar = iter(dataloader), tqdm(range(len(dataloader)))
	model.eval()
	for step in tbar:
		dict_data = next(dataloader_iter)
		x, y = dict_data['x'].to(args_infer.device), dict_data['y'].to(args_infer.device)

		if args_train.dataset_name == "voc":
			h, w = y.shape[1:]
			pad_h = ceil(h / args_train.stride_total) * args_train.stride_total - x.shape[2]
			pad_w = ceil(w / args_train.stride_total) * args_train.stride_total - x.shape[3]
			x = F.pad(x, pad=(0, pad_w, 0, pad_h), mode='reflect')
			logits = model(x)[:, :, :h, :w]

		else:
			logits = model(x)
			#logits = logits["pred"]

		prob, pred = F.softmax(logits.detach(), dim=1), logits.argmax(dim=1)

		metrics.update(y.cpu().numpy(), pred.cpu().numpy())
		running_score.update(y.cpu().numpy(), pred.cpu().numpy())

		ent, lc, ms, = [getattr(UncertaintySampler, strategy_method)(prob)[0].cpu()
						for strategy_method in ["_entropy", "_least_confidence", "_margin_sampling"]]
		dict_tensors = {'input': dict_data['x'][0].cpu(),
						'target': dict_data['y'][0].cpu(),
						'pred': pred[0].detach().cpu(),
						'confidence': lc,
						'margin': -ms,  # minus sign is to draw smaller margin part brighter
						'entropy': ent}
		
		# Save results
		try:
			img_name = os.path.basename(dataloader.dataset.images[step])[:-4]
		except:
			img_name = str(step)

		cv2.imwrite(os.path.join(args_infer.results_dir, img_name+"_p.png"), pred.cpu().numpy()[0])
		cv2.imwrite(os.path.join(args_infer.results_dir, img_name+"_l.png"), y.cpu().numpy()[0])
		
		'''if args_infer.save_grid:
			vis(dict_tensors, fp=f"{args_infer.results_dir}/{img_name}_grid.png")
		else:
			vis(dict_tensors, fp=f"{args_infer.results_dir}/{img_name}.png", grid=False)
		list_results.append(dict_tensors)'''

	# Save images, labels, predictions and querys into pickles
	if args_infer.save_pickle:
		pkl.dump(np.asarray([x['input'].numpy() for x in list_results]), open(f"{args_infer.results_dir}/inputs.pkl", "wb"))
		pkl.dump(np.asarray([x['target'].numpy() for x in list_results]), open(f"{args_infer.results_dir}/target.pkl", "wb"))
		pkl.dump(np.asarray([x['confidence'].numpy() for x in list_results]), open(f"{args_infer.results_dir}/qconfidences.pkl", "wb"))
		pkl.dump(np.asarray([x['margin'].numpy() for x in list_results]), open(f"{args_infer.results_dir}/qmargin.pkl", "wb"))
		pkl.dump(np.asarray([x['entropy'].numpy() for x in list_results]), open(f"{args_infer.results_dir}/qentropy.pkl", "wb"))
	

	# Save metrics into json file
	with open(f"{args_infer.results_dir}/metrics.json", 'w') as f:
		scores = running_score.get_scores()[0]
		miou, iou = scores['Mean IoU'], scores['All IoU']
		if args_train.dataset_name == "metalDAM":
			miou = np.mean(np.delete(iou, 3))  # Remove class 3
		metrics.metrics_dict['pp_miou'] = miou
		metrics.metrics_dict['pp_all_iou'] = iou.tolist()
		json.dump(metrics.metrics_dict, f)


@torch.no_grad()
def main(args_infer):
	with open(os.path.join(args_infer.model_dir, "args.json"), 'r') as j:
		args_train = json.loads(j.read())
		args_train = Namespace(**args_train)
	if args_train.n_pixels_by_us != 0:
		base_path_results = f"{args_infer.model_dir}/{str(args_infer.n_query)}_query/"
	else:
		base_path_results = f"{args_infer.model_dir}/fully_sup/"

	# Load data
	dataloader = get_dataloader(deepcopy(args_train), batch_size=1, n_workers=args_train.n_workers,
								val=True, test=True, query=False, shuffle=False)

	# Inference with best model chosen from accuracy on validation data
	args_infer.results_dir = os.path.join(base_path_results, "evaluate_train_model/")
	if not os.path.exists(args_infer.results_dir):
		os.makedirs(args_infer.results_dir)
	model_val = torch.load(os.path.join(base_path_results, "best_loss_model_train.pt"))
	evaluate(args_train, args_infer, dataloader, model_val)

	# Inference with best model chosen from accuracy on validation data
	args_infer.results_dir = os.path.join(base_path_results, "evaluate_val_model/")
	if not os.path.exists(args_infer.results_dir):
		os.makedirs(args_infer.results_dir)
	model_train = torch.load(os.path.join(base_path_results, "best_miou_model.pt"))
	evaluate(args_train, args_infer, dataloader, model_train)


if __name__ == '__main__':
	parser = ArgumentParser("Inference")
	parser.add_argument("--model_dir", type=str, required=True, help="Directory of the trained model")
	parser.add_argument("--n_query", type=int,  help="N query to evaluate")
	parser.add_argument('--save_grid', dest='save_grid', action='store_true', default=False,
						help="Save results as a grid of images")
	parser.add_argument('--no-save_grid', dest='save_grid', action='store_false', help="Save results individually")
	parser.add_argument("--save_pickle", action='store_true', default=False, help="If True, save results on pickle files")
	args = parser.parse_args()

	args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
	main(args)
