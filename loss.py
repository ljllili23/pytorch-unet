import torch
import torch.nn as nn

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()


# def dice_loss(pred, target):
#     """This definition generalize to real valued pred and target vector.
# This should be differentiable.
#     pred: tensor with first dimension as batch
#     target: tensor with first dimension as batch
#     """
#
#     smooth = 1.
#
#     # have to use contiguous since they may from a torch.view op
#     iflat = pred.contiguous().view(-1)
#     tflat = target.contiguous().view(-1)
#     intersection = (iflat * tflat).sum()
#
#     A_sum = torch.sum(tflat * iflat)
#     B_sum = torch.sum(tflat * tflat)
#
#     return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))