import sys
import time
import torch
with open('data/PhiBar.dat', 'rb') as file:
    Phi:torch.Tensor = torch.load(file)
print(Phi.pow(2).sum(-1), Phi.shape)