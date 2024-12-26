import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
import torch.distributions as dist
from torch.distributions.dirichlet import Dirichlet

class Entropy_Loss(nn.Module):
    def __init__(self, lamb=0.5):
        super(Entropy_Loss, self).__init__()
        self.lamb = lamb

    def forward(self, ID_output, OOD_output, ID_label):
        cls_loss = F.cross_entropy(ID_output, ID_label)
        ood_loss = -(OOD_output.mean(1) - torch.logsumexp(OOD_output, dim=1)).mean()
        return cls_loss + self.lamb * ood_loss

class Energy_Loss(nn.Module):
    def __init__(self, m_in=-7, m_out=-23, lamb=0.1):
        super(Energy_Loss, self).__init__()
        self.lamb = lamb
        self.m_in = m_in
        self.m_out = m_out

    def forward(self, ID_output, OOD_output, ID_label):
        clf_loss = F.cross_entropy(ID_output, ID_label)
        Ec_out = -torch.logsumexp(OOD_output, dim=1)
        Ec_in = -torch.logsumexp(ID_output, dim=1)
        energy_loss = self.lamb * (torch.pow(F.relu(Ec_in-self.m_in), 2).mean() + torch.pow(F.relu(self.m_out-Ec_out), 2).mean())
        loss = clf_loss + energy_loss
        return loss

class DPN_Loss(nn.Module):
    def __init__(self, target_alpha=15, lamb=0.1):
        super(DPN_Loss, self).__init__()
        self.lamb = lamb
        self.target_alpha = 15
    
    def target_2_alpha(target, num_classes=10):
        alpha = torch.ones((target.shape[0], num_classes)).cuda()
        alpha.scatter_(1, target.unsqueeze(1), self.target_alpha)

        return alpha

    def forward(self, ID_output, OOD_output, ID_label):
        num_classes = ID_output.shape[1]
        clf_loss = F.cross_entropy(ID_output, ID_label)
        ID_alpha = F.relu(ID_output)+1
        OOD_alpha = F.relu(OOD_output)+1
        ID_predicted_dirichlet = Dirichlet(ID_alpha)
        ID_target_dirichlet = Dirichlet(self.target_2_alpha(ID_label, num_classes=num_classes))
        clf_loss += torch.mean(dist.kl.kl_divergence(ID_predicted_dirichlet, ID_target_dirichlet))
        OOD_target = torch.ones((OOD_output[0].shape[0], num_classes)).cuda()
        OOD_predicted_dirichlet = Dirichlet(OOD_alpha)
        OOD_target_dirichlet = Dirichlet(OOD_target)
        reg_loss = self.lamb * torch.mean(dist.kl.kl_divergence(OOD_predicted_dirichlet, OOD_target_dirichlet))
        loss = clf_loss + reg_loss
        
        return loss


class DUL_Loss(nn.Module):
    def __init__(self, dul_m_in=-7, dul_m_out=-23, lamb=0.1, gamma=1.0, tau=2):
        super(DUL_Loss, self).__init__()
        self.lamb = lamb
        self.gamma = gamma
        self.tau = tau
        self.dul_m_in = dul_m_in
        self.dul_m_out = dul_m_out

    def diff_entropy(self, alphas):
        alpha0 = torch.sum(alphas, dim=1)
        return torch.sum(
            torch.lgamma(alphas)-(alphas-1)*(torch.digamma(alphas)-torch.digamma(alpha0).unsqueeze(1)),
            dim=1) - torch.lgamma(alpha0)

    def forward(self, ID_output, OOD_output, ID_label, original_ID_output, original_OOD_output):
        ID_alpha = F.relu(ID_output) + 1
        OOD_alpha = F.relu(OOD_output) + 1
        original_ID_alpha = F.relu(original_ID_output) + 1
        original_OOD_alpha = F.relu(original_OOD_output) + 1
        # original classification loss
        clf_loss = F.cross_entropy(ID_output, ID_label)
        
        # differential entropy
        h_in = self.diff_entropy(ID_alpha)
        h_out = self.diff_entropy(OOD_alpha)
        #original_h_in = self.diff_entropy(original_ID_alpha)
        #original_h_out = self.diff_entropy(original_OOD_alpha)
        original_h_in = 0
        original_h_out = 0

        # OOD detection loss
        diff_entropy_loss = self.lamb * (torch.pow(F.relu(h_in - (original_h_in + self.dul_m_in)), self.tau).mean() + \
                                        torch.pow(F.relu((original_h_out + self.dul_m_out) - h_out), self.tau).mean())
        #print(diff_entropy_loss)
        # unchanged overall uncertainty regularization term
        dul_reg = self.gamma * F.kl_div(F.log_softmax(OOD_output, dim=1), F.softmax(original_OOD_output, dim=1), reduction='batchmean')

        return clf_loss + diff_entropy_loss + dul_reg