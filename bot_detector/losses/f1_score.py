import torch
import torch.nn as nn
from torchmetrics import F1Score

from .build_loss import register_loss


@register_loss
class F1(nn.Module):
    def __init__(self, pred_name, target_name, threshold,
                 display_name=None, display_tqdm=False):
        super(F1, self).__init__()
        self.pred_name = pred_name
        self.target_name = target_name
        self.loss_name = display_name

        self.metric = F1Score(
            threshold=threshold, num_classes=1
        )

        self.preds = []
        self.targets = []
    
    def forward(self, container):
        y_pred = container[self.pred_name].float()
        y_pred = torch.sigmoid(y_pred)
        y_true = container[self.target_name].float()

        self.preds.append(y_pred.detach().cpu())
        self.targets.append(y_true.detach().cpu())
    
    def calc_metric(self):
        f1_score = self.metric(
            torch.cat(self.preds),
            torch.cat(self.targets).to(torch.int)
        )
        return {
            'F1_score': f1_score,
        }
    
    def reset(self):
        self.preds = []
        self.targets = []