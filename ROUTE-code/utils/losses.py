import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion_ce = nn.CrossEntropyLoss()

    def forward(self, outputs_p_w, outputs_u_w, targets_p, targets_u=None, targets_u_true=None):
        logits = torch.cat((outputs_p_w, outputs_u_w))
        labels = torch.cat((targets_p, targets_u_true))
        return self.criterion_ce(logits, labels)

class PULoss(nn.Module):
    def __init__(self, p_prior = 0.5, loss_type = "uPU"):
        super().__init__()
        self.p_prior = p_prior
        self.base_loss = nn.CrossEntropyLoss()
        self.loss_type = loss_type


    def forward(self, outputs_p_w, outputs_u_w, targets_p, targets_u):
        # R_p^+
        loss_pos_pos = self.p_prior * self.base_loss(outputs_p_w, targets_p)
        # R_u^_
        loss_unlabel_neg = self.base_loss(outputs_u_w, targets_u)
        # R_p^_
        targets_p_reverse = (1 - targets_p).to(targets_p.dtype)
        loss_pos_neg = - self.p_prior * self.base_loss(outputs_p_w, targets_p_reverse)

        if self.loss_type == "uPU":
            return loss_pos_pos + loss_unlabel_neg + loss_pos_neg
        elif self.loss_type == "nnPU":
            self.lda = torch.tensor([0.0]).to(outputs_p_w.device)
            return loss_pos_pos + torch.max(self.lda, loss_unlabel_neg + loss_pos_neg)
        else:
            ValueError('Error with loss!')









