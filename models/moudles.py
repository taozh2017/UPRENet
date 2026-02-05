import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import torch.nn.functional as F
# from einops.layers.torch import Rearrange
from torch import nn, einsum
# from einops import rearrange

class Decoder(nn.Module):
    def __init__(self, in_dim=100):
        super(Decoder, self).__init__()
        self.in_dim = in_dim
        
        self.res1 = nn.Sequential(
            nn.Conv2d(self.in_dim + 1, self.in_dim, kernel_size=1, padding=0, bias=False),
            # nn.BatchNorm2d(100),
            nn.ReLU(inplace=True))
        self.res2 = nn.Sequential(
            nn.Conv2d(self.in_dim, self.in_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_dim, self.in_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True))
        self.down = nn.Sequential(
            nn.Conv2d(self.in_dim, 100, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(100, 1, kernel_size=1, padding=0, bias=False))
        self.softplus = nn.Softplus()

    def forward(self, x):

        x = self.res1(x)
        x = self.res2(x) + x
        out = self.down(x)

        return out

    

class RDF(nn.Module):
    def __init__(self, c=2):
        super(RDF, self).__init__()
        self.conv = nn.Conv2d(in_channels=2*c, out_channels=c, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, predition_1, predition_2, u):
        u_ = u[0].view(256, 256, 1).permute(2, 0, 1).unsqueeze(0)
        u_en = u[1].view(256, 256, 1).permute(2, 0, 1).unsqueeze(0)

        u1 = u_.expand_as(predition_1)
        u2 = u_en.expand_as(predition_2)
        
        epsilon = 1e-6
        
        # Incorporate uncertainty as weights
        W1 = 1 / (u1 + epsilon)
        W2 = 1 / (u2 + epsilon)

        # Compute fused probabilities
        predition = (W1 * predition_1 + W2 * predition_2) / (W1 + W2 + epsilon)
        p1 = (W1 * predition_1)/(W1 + W2 + epsilon)
        p2 = (W2 * predition_2)/(W1 + W2 + epsilon)
        
        weight = torch.cat([p1, p2], dim=1)
        G = self.conv(weight)
        G = self.softmax(G).squeeze(0)
        # G = self.sigmoid(G).squeeze(0)
        
        predition = G[0] * p1 + G[1] * p2
        
        return predition