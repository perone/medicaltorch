import torch.nn as nn


def dice_loss(input, target):
    eps = 0.0001

    iflat = input.view(-1)
    tflat = target.view(-1)

    intersection = (iflat * tflat).sum()
    union = iflat.sum() + tflat.sum()

    dice = (2.0 * intersection + eps) / (union + eps)

    return - dice


class MaskedDiceLoss(nn.Module):
    def __init__(self, ignore_value=-100.0):
        super().__init__()
        self.ignore_value = ignore_value

    def forward(self, input, target):
        eps = 0.0001

        masking = target == self.ignore_index
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
