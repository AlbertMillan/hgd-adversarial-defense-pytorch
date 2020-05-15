import torch
import torch.nn as nn
import numpy as np
import sys

from denoiser import Denoiser


# class DenoiseLoss(nn.Module):
#     def __init__(self, n, hard_mining=0, norm=False):
#         super(DenoiseLoss, self).__init__()
        
#         self.n = n
#         assert(hard_mining >= 0 and hard_mining <= 1)
#         self.hard_mining = hard_mining
#         self.norm = norm
        
#     def forward(self, x, y):
#         loss = torc


# class Loss(nn.Module):
#     def __init__(self, n, hard_mining=0, norm=False):
#         super(Loss, self).__init__()
# #         self.loss = DenoiseLoss(n, hard_mining, norm)
#         self.n = n
        
#     def forward(self, out_adv, out_org):
#         z = []
#         for i in range(len(out_adv)):
#             loss = torch.pow(torch.abs(out_adv[i] - out_org[i]), self.n) / self.n
#             loss = loss.mean()
#             z.append(loss)

# #         print("LOSS:", z)
# #         sys.exit()
#         return torch.stack(z).mean()


class FullDenoiser(nn.Module):
    
    def __init__(self, target_model, n=1, hard_mining=0, loss_norm=False):
        super(FullDenoiser, self).__init__()
        
        # 1. Load Models
        self.target_model = target_model
        self.denoiser = Denoiser(x_h=32, x_w=32)
        
#         self.loss = Loss(n, hard_mining=0, norm=loss_norm)
        
        
    def _no_grad_step(self, x_batch):
        """ Performs a step during testing."""
        logits = None
        with torch.no_grad():
            logits = self.target_model(x_batch)
            
        return logits
        
        
    def forward(self, x, x_adv):
        
        # 1. Compute denoised image. Need to check this...
        noise = self.denoiser.forward(x_adv)
        x_smooth = x_adv + noise
        
        # 2. Get logits from smooth and denoised image
        out_adv = self.target_model(x_smooth)
        out_org = self.target_model(x)
#         out_adv = self._no_grad_step(x_smooth, y)
#         out_org = self._no_grad_step(, y)

        loss = torch.sum(torch.abs(out_adv - out_org), dim=1).mean()
#         loss = torch.mean(torch.abs(out_adv - out_org)
    
        return out_adv, out_org, loss