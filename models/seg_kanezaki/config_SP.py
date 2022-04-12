from easydict import EasyDict as edict


__C = edict()
# Consumers can get config by: from config import cfg

cfg = __C

__C.TRAIN_CONFIGURATION = 'base_folder'  # available options : base, base_folder,
__C.DATA_INFO = 'floodnet'  # available options : base, base_folder, ref
__C.OPTUNA = True  # train with optuna
__C.OPTIMIZER = "sgd"  # Optimizers: adam, sgd, asgd
__C.CKPT = ''  # Path to base ckp. Empty if training from scratch


# Options common to all train_configurations
__C.COMMON = edict()

__C.COMMON.CROP_BAR_SZ = 65  # crop size for image botton bar
__C.COMMON.CLAHE = False  # apply clahe
__C.COMMON.DATA_AUGM = False  # apply random clahe data augmentation
__C.COMMON.BLUR = False  # apply clahe
__C.COMMON.NCHANNEL = 100  # number of channels
__C.COMMON.MAXITER = 2000  # number of maximum iterations
__C.COMMON.MINLABELS = None  # minimum number of labels
__C.COMMON.LR = 0.1
__C.COMMON.NCONV = 5  # N convolutions of kanezaki CNN. Ignored for other architectures
__C.COMMON.SUPERPIXEL_ALGO = "Felzenszwalb"  # Superpixel algorithms: SLIC, Felzenszwalb
__C.COMMON.NUM_SUPERPIXELS = 4000
__C.COMMON.COMPACTNESS_VALUE = 20
__C.COMMON.FZ_SIGMA = 0.8
__C.COMMON.FZ_SCALE = 10
__C.COMMON.FZ_MIN_SIZE = 10


# Base train_configuration options
__C.BASE = edict()

__C.BASE.IMG_PATH = ''
__C.BASE.DIR_LABEL_PATH = ''
__C.BASE.PRED_SAVE_PATH = '../../experiments/floodnet/'


# Base_folder train_configuration options
__C.BASE_FOLDER = edict()

__C.BASE_FOLDER.DIR_PATH = '../../../../data/floodnet/n20_train_images/'
__C.BASE_FOLDER.DIR_LABEL_PATH = '../../../../data/floodnet/n20_train_masks/'
__C.BASE_FOLDER.DIR_SAVE_PATH = '../../experiments/floodnet/USSP/'


# Optuna types
__C.OPTUNA_T = edict()

__C.OPTUNA_T.DATA_AUGM = "categorical"
__C.OPTUNA_T.OPTIMIZER = "categorical"
__C.OPTUNA_T.NCHANNEL = "int"
__C.OPTUNA_T.MAXITER = "int"
__C.OPTUNA_T.LR = "categorical"
__C.OPTUNA_T.NCONV = "int"
__C.OPTUNA_T.NUM_SUPERPIXELS = "int"
__C.OPTUNA_T.COMPACTNESS_VALUE = "int"
__C.OPTUNA_T.FZ_SCALE = 'int'
__C.OPTUNA_T.FZ_MIN_SIZE = 'int'

# Optuna values
__C.OPTUNA_V = edict()

__C.OPTUNA_V.N_TRIALS = 20

__C.OPTUNA_V.DATA_AUGM = [True, False]
__C.OPTUNA_V.OPTIMIZER = ['adam', 'sgd', 'asgd']
__C.OPTUNA_V.NCHANNEL = [25, 75, 25]
__C.OPTUNA_V.MAXITER = [1000, 3000, 2000]
__C.OPTUNA_V.LR = [0.005, 0.01, 0.05, 0.1]
__C.OPTUNA_V.NCONV = [2, 6, 1]
__C.OPTUNA_V.NUM_SUPERPIXELS = [2000, 4000, 6000]
__C.OPTUNA_V.COMPACTNESS_VALUE = [15, 20, 25]
__C.OPTUNA_V.FZ_SCALE = [1, 10, 100]
__C.OPTUNA_V.FZ_MIN_SIZE = [10, 30, 100]