from easydict import EasyDict as edict


__C = edict()
# Consumers can get config by: from config import cfg

cfg = __C

__C.TRAIN_CONFIGURATION = 'ref'  # available options : base, base_folder, ref
__C.DATA_INFO = 'metaldam'  # available options : base, base_folder, ref
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
__C.COMMON.MAXITER = 2  # number of maximum iterations
__C.COMMON.MINLABELS = 5  # minimum number of labels
__C.COMMON.LR = 0.1
__C.COMMON.NCONV = 5  # N convolutions of kanezaki CNN. Ignored for other architectures
__C.COMMON.STEPSIZE_SIM = 1  # step size for similarity loss
__C.COMMON.STEPSIZE_CON = 1  # step size for continuity loss
__C.COMMON.STEPSIZE_SCR = 1  # step size for scribble loss


# Base train_configuration options
__C.BASE = edict()

__C.BASE.IMG_PATH = ''
__C.BASE.DIR_LABEL_PATH = '/'
__C.BASE.PRED_SAVE_PATH = '../../models/test/'
__C.BASE.SCRIBBLE_PATH = ''


# Base_folder train_configuration options
__C.BASE_FOLDER = edict()

__C.BASE_FOLDER.DIR_PATH = '../../../../data/metaldam/n20_train_images/'
__C.BASE_FOLDER.DIR_LABEL_PATH = '../../../../data/metaldam/n20_train_masks/'
__C.BASE_FOLDER.DIR_SAVE_PATH = '../../models/metaldam/SCR/'
__C.BASE_FOLDER.DIR_SCRIBBLE_PATH = '../../../../data/metaldam/n20_train_scribbles/'


# Ref train_configuration options
__C.REF = edict()

__C.REF.TRAIN_DIR_PATH = '../../../../data/metaldam/n20_train_images/'
__C.REF.TEST_DIR_PATH = '../../../../data/metaldam/n20_train_images/'
__C.REF.DIR_LABEL_PATH = '../../../../data/metaldam/n20_train_masks/'
__C.REF.DIR_SAVE_PATH = '../../models/metaldam/USCRef/'
__C.REF.DIR_SCRIBBLE_PATH = ''  # load scribbles if not empty
__C.REF.MAXUPDATE = 20  # number of maximum update count
__C.REF.BATCH_SIZE = 1  # number of batch_size


# Optuna types
__C.OPTUNA_T = edict()

__C.OPTUNA_T.DATA_AUGM = "categorical"
__C.OPTUNA_T.OPTIMIZER = "categorical"
__C.OPTUNA_T.NCHANNEL = "int"
__C.OPTUNA_T.MAXITER = "int"
__C.OPTUNA_T.LR = "categorical"
__C.OPTUNA_T.NCONV = "int"
__C.OPTUNA_T.STEPSIZE_SIM = "float"
__C.OPTUNA_T.STEPSIZE_CON = "float"
__C.OPTUNA_T.STEPSIZE_SCR = "float"


# Optuna values
__C.OPTUNA_V = edict()

__C.OPTUNA_V.N_TRIALS = 20

__C.OPTUNA_V.DATA_AUGM = [True, False]
__C.OPTUNA_V.OPTIMIZER = ['adam', 'sgd', 'asgd']
__C.OPTUNA_V.NCHANNEL = [50, 150, 25]
__C.OPTUNA_V.MAXITER = [200, 300, 20]
__C.OPTUNA_V.LR = [0.001, 0.005, 0.01, 0.05, 0.1]
__C.OPTUNA_V.NCONV = [2, 12, 2]
__C.OPTUNA_V.STEPSIZE_SIM = [0.1, 1.0, 0.2]
__C.OPTUNA_V.STEPSIZE_CON = [0.1, 1.0, 0.2]
__C.OPTUNA_V.STEPSIZE_SCR = [0.1, 1.0, 0.2]
