

import torch
import torch.nn as nn
import torchvision

class Binarize(nn.Module):

  def forward(self, x):
    new_x = torch.zeros_like(x)
    new_x[x > 0.5] = 1.0
    new_x[x <= 0.5] = 0.0
    return new_x
  
class PutOnDevice(nn.Module):
  
  def __init__(self, device):
    super().__init__()
    self.device = device
  
  def forward(self, x):
    return x.to(self.device)
  
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

def get_training_data(device):
  train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', 
                             train=True,
                             download=True,
                             transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                Binarize(),
                                PutOnDevice(device),
                              ])),
                             batch_size=128, 
                             shuffle=True)
  return train_loader
def get_testing_data(device):
  test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', 
                             train=False, 
                             download=True,
                             transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                Binarize(),
                                PutOnDevice(device),
                             ])),
                             batch_size=128, 
                             shuffle=True)
  return test_loader
