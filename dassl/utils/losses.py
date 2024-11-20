import torch
import torch.nn as nn

__all__ = ["Consistency_loss"]

class Consistency_loss(nn.Module):
    
    def __init__(self):
        super(Consistency_loss, self).__init__()

    def forward(self, x):
        # x = x[:, 1:] # B T(8) D(512)

        x = x/x.norm(dim=2).unsqueeze(-1)
        mtx = x @ x.permute(0, 2, 1)
        bm_mtx = mtx.mean(dim=0) # average the batch
        loss = -torch.log((bm_mtx+1)*(0.5)).mean()

        return loss
