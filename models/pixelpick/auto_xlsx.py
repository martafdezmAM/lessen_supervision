import os
import json
import glob
import pandas as pd
from argparse import ArgumentParser, Namespace


def main(args):

	with open(os.path.join(args.model_dir, "args.json"), 'r') as j:
		args_train = json.loads(j.read())
		args_train = Namespace(**args_train)
		
	list_folders = sorted(glob.glob(os.path.join(args.model_dir, "*/")))
	
	model_metrics = []
	for mode in ["train", "val"]:
		try:
			for folder in list_folders:
				if args_train.n_pixels_by_us != 0:
					base_path_results = folder
				else:
					base_path_results = f"{args.model_dir}/fully_sup"
				results_dir = os.path.join(base_path_results, "evaluate_" + mode + "_model/")
				with open(os.path.join(results_dir, "metrics.json"), 'r') as f:
					metrics = json.loads(f.read())
					
				pp_metrics = {}
				pp_metrics["mode"] = mode
				pp_metrics["folder"] = os.path.basename(base_path_results)
				pp_metrics["pp_miou"] = metrics["pp_miou"]
				for c in range(len(metrics["pp_all_iou"])):
					pp_metrics[str(c)+"_iou"] = metrics["pp_all_iou"][c]
				model_metrics.append(pp_metrics)
		except:
			print("{} note found".format(mode))
	df = pd.DataFrame(model_metrics)
	print(df)
	df.to_excel(os.path.join(args.model_dir, "metrics.xlsx"))


if __name__ == '__main__':
	parser = ArgumentParser("Inference")
	parser.add_argument("--model_dir", type=str, required=True, help="Directory of the trained model")
	args = parser.parse_args()

	main(args)


def colour_images(predictions: np.ndarray, colors =[[255, 255, 0],[0, 0, 255],[21, 180, 214],[75, 179, 36],[75, 179, 36]]) -> list:
    list_rgb = []
    colors = np.array(colors)/256.0
    for i in range(predictions.shape[0]):
        #predLabels = np.argmax(predictions[i], axis=-1)
        rgb_predLabels = np.zeros(shape=(predictions[i].shape[0], predictions[i].shape[1],3))
        for j in range(len(colors)):
            indicesJ = (predictions[i] == j)
            rgb_predLabels[indicesJ] = colors[j]
        list_rgb.append(rgb_predLabels)
    return list_rgb