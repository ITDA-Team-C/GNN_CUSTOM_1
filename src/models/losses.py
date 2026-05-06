import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        if self.pos_weight is not None:
            return F.binary_cross_entropy_with_logits(
                logits, targets.float(), pos_weight=self.pos_weight
            )
        else:
            return F.binary_cross_entropy_with_logits(logits, targets.float())


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        pt = torch.sigmoid(logits)
        targets = targets.float()

        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        modulation = (1 - pt) ** self.gamma
        focal_loss = self.alpha * modulation * ce_loss

        return focal_loss.mean()
