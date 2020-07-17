

import torch
import torch.nn as nn


def build_model(device):
  model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=5),
    nn.MaxPool2d(2),
    nn.Tanh(),
    nn.Conv2d(16, 32, kernel_size=5),
    nn.MaxPool2d(2),
    nn.Tanh(),
    nn.Flatten(),
    nn.Linear(512, 64),
    nn.Tanh(),
    nn.Linear(64, 10),
    nn.Softmax(dim=1),
  )
  model = model.to(device)
  return model
