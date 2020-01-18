import torch
from torch.nn import Module

def dice_loss(input, target):
    """Dice loss.

    :param input: The input (predicted)
    :param target:  The target (ground truth)
    :returns: the Dice score between 0 and 1.
    """
    eps = 0.0001

    iflat = input.view(-1)
    tflat = target.view(-1)

    intersection = (iflat * tflat).sum()
    union = iflat.sum() + tflat.sum()

    dice = (2.0 * intersection + eps) / (union + eps)

    return - dice


class MaskedDiceLoss(Module):
    """A masked version of the Dice loss.

    :param ignore_value: the value to ignore.
    """

    def __init__(self, ignore_value=-100.0):
        super().__init__()
        self.ignore_value = ignore_value

    def forward(self, input, target):
        eps = 0.0001

        masking = target == self.ignore_value
        masking = masking.sum(3).sum(2)
        masking = masking == 0
        masking = masking.squeeze()

        labeled_target = target.index_select(0, masking.nonzero().squeeze())
        labeled_input = input.index_select(0, masking.nonzero().squeeze())

        iflat = labeled_input.view(-1)
        tflat = labeled_target.view(-1)

        intersection = (iflat * tflat).sum()
        union = iflat.sum() + tflat.sum()

        dice = (2.0 * intersection + eps) / (union + eps)

        return - dice


class ConfidentMSELoss(Module):
    def __init__(self, threshold=0.96):
        self.threshold = threshold
        super().__init__()

    def forward(self, input, target):
        n = input.size(0)
        conf_mask = torch.gt(target, self.threshold).float()
        input_flat = input.view(n, -1)
        target_flat = target.view(n, -1)
        conf_mask_flat = conf_mask.view(n, -1)
        diff = (input_flat - target_flat)**2
        diff_conf = diff * conf_mask_flat
        loss = diff_conf.mean()
        return loss