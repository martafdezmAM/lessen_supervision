# Unsupervised and weak-learning semantic segmentation methods
A library of unsupervised learning and weak-learning approaches for semantic segmentation tasks.

## 1. Getting started
First install on your python environment the package with:
```bash
    cd UnsupervisedSeg
    pip install .
```
In case you want to modify the library and run it with your local changes:
```bash
    cd UnsupervisedSeg
    pip install -e .
```

## 2. Unsupervised methods
### 2.1 IIC - Invariant Information Clustering
[Paper](https://arxiv.org/abs/1807.06653) [Code](https://github.com/xu-ji/IIC)
#### Train:
1. Create/modify an ".config" file with the configuration that will be used for training the model
2. Run training script:
```bash
    cd models/iic/code/scripts/
    python segmentation_twohead_launch.py --config path_config_file
```
#### Train with hyper-parameter optimization:
We use Optuna hyperparameter optimization framework to automate hyperparameter search by running:
```bash
    cd models/iic/code/scripts/
    python segmentation_twohead_launch.py --config path_config_file --optuna
```
### 2.2 Unsupervised image segmentation by backpropagation
[Paper](https://kanezaki.github.io/pytorch-unsupervised-segmentation/ICASSP2018_kanezaki.pdf) [Code](https://github.com/kanezaki/pytorch-unsupervised-segmentation)
#### Train
1. Modify *models/seg_kanezaki/config_SP.py* file with the configuration that will be used for training the model
2. Run training script:
```bash
    cd models/seg_kanezaki/code/scripts/
    python train.py --method sp
```

### 2.3 Unsupervised Learning of Image Segmentation Based on Differentiable Feature Clustering
[Paper](https://arxiv.org/abs/2007.09990) [Code](https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip)
#### Train
1. Modify *models/seg_kanezaki/config_ST.py* file with the configuration that will be used for training the model
2. Run training script:
```bash
    cd models/seg_kanezaki/code/scripts/
    python train.py --method st
```


## 3. Weakly-supervised methods
### 3.1 Scribbles: Unsupervised Learning of Image Segmentation Based on Differentiable Feature Clustering
[Paper](https://arxiv.org/abs/2007.09990) [Code](https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip)
#### Train
1. Modify *models/seg_kanezaki/config_ST.py* file **includding the paths where the scribbles are stored**
2. Run training script:
```bash
    cd models/seg_kanezaki/code/scripts/
    python train.py --method st
```
