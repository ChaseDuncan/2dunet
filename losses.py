import torch
import torch.nn as nn

class BraTSBCEWithLogitsLoss(nn.Module):
    def __init__(self):
       super(BraTSBCEWithLogitsLoss, self).__init__()
       self.loss_et    = nn.BCEWithLogitsLoss()
       self.loss_wt    = nn.BCEWithLogitsLoss()
       self.loss_tc    = nn.BCEWithLogitsLoss()

    def forward(self, output, target):
        loss_sum    = self.loss_et(output, target)\
                    + self.loss_wt(output, target)\
                    + self.loss_tc(output, target)

        return loss_sum

