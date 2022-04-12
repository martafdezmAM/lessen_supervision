# Seg_kanezaki

Implementation of the most recent unsupervised segmentation contributions of Asako Kanezaki.


## Superpixel refinement: SP method

![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

Implementation of the paper 
["Unsupervised microstructure segmentation by mimicking metallurgists approach to pattern recognition"](https://www.nature.com/articles/s41598-020-74935-8)
that applies the unsupervised method 
["Unsupervised Image Segmentation by Backpropagation"](https://www.nature.com/articles/s41598-020-74935-8) introduced by Kanezaki.
- ["Unsupervised Image Segmentation by Backpropagation" code repository](https://github.com/kanezaki/pytorch-unsupervised-segmentation)


### Training 
This method has 2 training configurations:
- Train and return otuput for a single image
- Train and return output for all the images at a folder. **Each image is trained with a different model**.

The train configuration and the hyperparameters can be modified at config_SP.py
```bash
python train.py --method sp
```

## Segmention with continuity loss and scribble alternative: ST method
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

Implementation of the paper "Unsupervised Learning of Image Segmentation Based on Differentiable Feature Clustering":
- [Original paper](https://arxiv.org/abs/2007.09990)
- [Original repository ](https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip)

This paper is an extension of the SP method. It includes:
- Better performance with spatial continuity loss
- Option of using scribbles as user input
- Option of using reference image(s)


### Training 
This method has 3 training configurations:
- Train and return otuput for a single image
- Train and return output for all the images at a folder. **Each image is trained with a different model**.
- Train the model with all the images from the training folder and returns the prediction for the testing images.

The train configuration and the hyperparameters can be modified at config_ST.py
```bash
python train.py --method st
```
