import torch
import torch.nn as nn
import torch.nn.functional as F

from .build_loss import register_loss


@register_loss
class BinaryCrossEntropy(nn.Module):
    def __init__(self, pred_name, target_name, loss_weight,
                 display_name=None, display_tqdm=False, smooth_factor=None,
                 neg_samples_weight=1, pos_samples_weight=1):
        super(BinaryCrossEntropy, self).__init__()
        self.pred_name = pred_name
        self.target_name = target_name
        self.loss_weight = loss_weight
        self.display_tqdm = display_tqdm
        self.smooth_factor = smooth_factor
        self.pos_samples_weight = pos_samples_weight
        self.neg_samples_weight = neg_samples_weight

        self.loss_name = display_name
        self.nested_loss = nn.BCELoss(reduction='none')
    
    def forward(self, container):
        y_pred = container[self.pred_name].float().view(-1)
        y_true = container[self.target_name].float().view(-1)

        y_pred = y_pred[y_true >= 0]
        y_true = y_true[y_true >= 0]

        pos_sample_mask = y_true > 0
        if self.smooth_factor is not None:
            targets = (1 - y_true) * self.smooth_factor + y_true * (1 - self.smooth_factor)
        else:
            targets = y_true

        loss = F.binary_cross_entropy_with_logits(
            y_pred, targets, reduction="none",
        )
        loss = torch.where(
            pos_sample_mask,
            loss * self.pos_samples_weight,
            loss * self.neg_samples_weight
        )
        loss = loss.mean()
        return loss