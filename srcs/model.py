import torch 
from torch import nn 

class model(nn.Module):
  def __init__(self):
    super(model, self).__init__()

    self.class_layar = nn.Sequential(
        nn.Flatten(),
        nn.Linear(32*32, 512),
        nn.ReLU(),
        nn.Linear(512,128),
        nn.ReLU(),
        nn.Linear(128,10))


  def forward(self, x):
    logits = self.class_layar(x)
    return logits

