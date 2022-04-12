import os
import json
import datetime
from argparse import ArgumentParser
import segmentation_models_pytorch as sm


sm_architectures = ["Unet", "UnetPlusPlus", "MAnet", "Linknet", "FPN", "PSPNet", "DeepLabV3", "DeepLabV3Plus", "PAN",]


class Arguments:
    def __init__(self):
        parser = ArgumentParser("PixelPick")

        parser.add_argument("--debug", "-d", action="store_true", default=False)
        parser.add_argument("--dir_root", type=str, default="..")
        parser.add_argument("--gpu_ids", type=str, nargs='+', default='0')
        parser.add_argument("--n_workers", type=int, default=0)
        parser.add_argument("--seed", "-s", type=int, default=0)
        parser.add_argument("--suffix", type=str, default='')
        
        # Model
        parser.add_argument("--network_name", type=str, default="DeepLabV3Plus", choices=sm_architectures)
        parser.add_argument("--backbone", type=str, default="efficientnet-b0", choices=sm.encoders.encoders.keys())
        parser.add_argument("--loss", type=str, default="ce", choices=["ce", "jaccard", "dice", "focal",])
        parser.add_argument("--continuity_weight", type=float, default=0.0, help="Weight of the continuty loss")

        # active learning
        parser.add_argument("--n_pixels_by_us", type=int, default=10, help="# pixels selected by a uncertainty sampling")
        parser.add_argument("--top_n_percent", type=float, default=0.05)
        parser.add_argument("--query_strategy", '-qs', type=str, default="margin_sampling",
                            choices=["least_confidence", "margin_sampling", "entropy", "random"])
        parser.add_argument("--reverse_order", action="store_true", default=False)

        # QBC
        parser.add_argument("--use_mc_dropout", action="store_true", default=False)
        parser.add_argument("--mc_dropout_p", type=float, default=0.2)
        parser.add_argument("--mc_n_steps", type=int, default=20)
        parser.add_argument("--vote_type", type=str, default="soft", choices=["soft", "hard"])

        # for supp. mat.
        parser.add_argument("--n_init_pixels", type=int, default=0, help="# pixels selected by a uncertainty sampling")
        parser.add_argument("--max_budget", type=int, default=100, help="maximum budget in pixels per image")
        parser.add_argument("--nth_query", type=int, default=1)

        # dataset
        parser.add_argument("--dataset_name", type=str, default="cv",
                            choices=["cs", "cv", "voc", "metalDAM", "flood", ])
        parser.add_argument("--dir_datasets", type=str, default="/scratch/shared/beegfs/gyungin/datasets")
        parser.add_argument("--downsample", type=int, default=4, help="downsample for Cityscapes training set")
        parser.add_argument("--use_aug", type=bool, default=True, help="data augmentation")
        parser.add_argument("--use_augmented_dataset", action="store_true", default=False,
                            help="whether to use the augmented dataset for pascal voc")

        self.parser = parser

    def parse_args(self):
        args = self.parser.parse_args()
        args.augmentations = {
            "geometric": {
                "random_scale": args.use_aug,
                "random_hflip": args.use_aug,
                "crop": args.use_aug
            },
            "photometric": {
                "random_color_jitter": args.use_aug,
                "random_grayscale": args.use_aug,
                "random_gaussian_blur": args.use_aug
            }
        }
        if args.dataset_name == "cs":
            args.batch_size = 4
            args.dir_dataset = "/scratch/shared/beegfs/gyungin/datasets/cityscapes"
            args.ignore_index = 19
            args.mean, args.std = [0.28689554, 0.32513303, 0.28389177], [0.18696375, 0.19017339, 0.18720214]
            args.n_classes = 19
            args.n_epochs = 50
            args.optimizer_type = "Adam"
            args.lr_scheduler_type = "Poly"
            assert args.lr_scheduler_type in ["Poly", "MultiStepLR"]
            # This params are for Adam
            args.optimizer_params = {
                "lr": 5e-4,
                "betas": (0.9, 0.999),
                "weight_decay": 2e-4,
                "eps": 1e-7
            }

        elif args.dataset_name == "cv":
            args.batch_size = 4
            args.dir_dataset = "../../data/camvid/"
            args.downsample = 1
            args.ignore_index = 11
            args.mean = [0.41189489566336, 0.4251328133025, 0.4326707089857]
            args.std = [0.27413549931506, 0.28506257482912, 0.28284674400252]
            args.n_classes = 11
            args.n_epochs = 50
            args.optimizer_type = "Adam"
            args.lr_scheduler_type = "MultiStepLR"
            assert args.lr_scheduler_type in ["Poly", "MultiStepLR"]
            # This params are for Adam
            args.optimizer_params = {
                "lr": 5e-4,
                "betas": (0.9, 0.999),
                "weight_decay": 2e-4,
                "eps": 1e-7
            }

        elif args.dataset_name == "voc":
            args.batch_size = 10
            args.dir_dataset = "/scratch/shared/beegfs/gyungin/datasets/VOC2012"
            args.dir_augmented_dataset = "/scratch/shared/beegfs/gyungin/datasets/VOC2012/VOCdevkit/VOC2012/train_aug"
            args.ignore_index = 255
            args.mean, args.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            args.n_classes = 21
            args.n_epochs = 50
            args.size_base = 400
            args.size_crop = 320
            args.optimizer_type = "SGD"
            args.lr_scheduler_type = "Poly"
            args.optimizer_params = {
                "lr": 1e-2,
                "weight_decay": 1e-4,
                "momentum": 0.9
            }

        elif args.dataset_name == "metalDAM":
            args.batch_size = 2
            args.dir_dataset = "../../data/metaldam/"
            args.splits = True
            args.query_path = '../../data/metaldam/metaldam_split.npy'
            args.dir_root = "experiments/"
            args.ignore_index = 255
            args.n_classes = 5
            args.crop_bar = 65
            args.n_epochs = 500
            args.optimizer_type = "Adam"
            args.lr_scheduler_type = "Poly"
            assert args.lr_scheduler_type in ["Poly", "MultiStepLR"]

            args.optimizer_params = {
                "lr": 0.01,
                "betas": (0.9, 0.999),
                "weight_decay": 1e-5,
            }

        elif args.dataset_name == "flood":
            args.batch_size = 2
            args.splits = True
            args.dir_dataset = "../../data/floodnet/"
            args.query_path = '../../data/floodnet/floodnet.npy'
            args.dir_root = "experiments/"
            args.ignore_index = -1
            args.n_classes = 10
            args.crop_bar = 0
            args.n_epochs = 500
            args.optimizer_type = "Adam"
            args.lr_scheduler_type = "Poly"
            assert args.lr_scheduler_type in ["Poly", "MultiStepLR"]

            args.optimizer_params = {
                "lr": 0.01,
                "betas": (0.9, 0.999),
                "weight_decay": 1e-4,
            }

        # naming
        list_keywords = list()
        list_keywords.append(args.dataset_name)
        list_keywords.append(f"d{args.downsample}") if args.dataset_name == "cs" else None
        list_keywords.append(args.network_name)

        # query strategy
        list_keywords.append(f"{args.query_strategy}") if args.n_pixels_by_us > 0 else None
        list_keywords.append(f"{args.vote_type}") if args.use_mc_dropout else None
        list_keywords.append(f"{args.n_pixels_by_us}")
        list_keywords.append(f"p{args.top_n_percent}") if args.top_n_percent > 0. and args.n_pixels_by_us > 0 else None
        list_keywords.append("reverse") if args.reverse_order else None

        list_keywords.append(str(args.seed))
        list_keywords.append(args.suffix) if args.suffix != '' else None
        list_keywords.append("debug") if args.debug else None

        try:
            args.experim_name = '_'.join(list_keywords)
        except TypeError:
            raise TypeError(list_keywords)

        # create dirs
        args.experim_name = args.experim_name + "_{}".format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        args.dir_checkpoints = f"{args.dir_root}/checkpoints/{args.experim_name}"
        os.makedirs(args.dir_checkpoints, exist_ok=True)

        with open(f"{args.dir_checkpoints}/args.json", 'w') as f:
            json.dump(vars(args), f)

        print(f"\nmodel name: {args.experim_name}\n")
        return args
