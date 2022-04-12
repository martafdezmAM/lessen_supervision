from typing import Optional

import torch
import torch.nn.functional as F
from segmentation_models_pytorch import losses
from segmentation_models_pytorch.utils import losses as basic_losses


class ContinuityLoss(torch.nn.Module):
    def __init__(self, initial_loss, stepsize_con: float = 1.0) -> None:
        super().__init__()
        self.stepsize_con = stepsize_con
        self.continuityLoss = torch.nn.L1Loss(size_average = True)
        self.initial_loss = initial_loss
        self.__name__ = initial_loss.__name__ + "_continuity"

    def forward(self, logits: torch.Tensor, target: torch.Tensor, ignore_index: float = 255.0):
        # continuity loss definition
        lhpy = self.continuityLoss(logits[:, :, 1:, :], logits[:, :, 0:-1, :])
        lhpz = self.continuityLoss(logits[:, :, :, 1:], logits[:, :, :, 0:-1])

        return (1 - self.stepsize_con) * self.initial_loss(logits, target, ignore_index=ignore_index) + self.stepsize_con * (lhpy + lhpz)


class CategoricalCrossEntropyLoss(basic_losses.CrossEntropyLoss):

    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, weight: Optional[torch.Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', mode=None) -> None:
        super(CategoricalCrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        #_target = torch.argmax(target, dim=1)
        return F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)
    

def custom_losses():
	my_dict = {"jaccard": losses.jaccard.JaccardLoss,
			   "dice": losses.dice.DiceLoss,
			   "ce": CategoricalCrossEntropyLoss,
			   "focal": losses.focal.FocalLoss,
			   "continuity": ContinuityLoss
			   }
	return my_dict
