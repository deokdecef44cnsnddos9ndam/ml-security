import torch
import torch.nn as nn
import torchvision

import mlsec.utils as ut

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

def get_examples(training_data):
  total = 0
  examples = []
  example_labels = []
  for data, labels in training_data:
    examples.append(data)
    example_labels.append(labels)
    total += len(data)
    if total > 200:
      break
      
  examples = torch.cat(examples)[:200].cpu()
  example_labels = torch.cat(example_labels)[:200].cpu()
  return examples, example_labels

def make_label(num, device):
  return (torch.ones((1)) * num).long().to(device)

def run_test(model, dataset):
  correct = 0.0
  total = 0.0

  for data, labels in dataset:
    # Get model outputs
    outputs = model(data).cpu()
    # Classify by whatever had the highest confidence
    pred = torch.argmax(outputs, dim=1)
    # Correct examples 
    correct_preds = pred == labels
    # Increment our counters
    correct += torch.sum(correct_preds)
    total += len(labels)

  accuracy = (correct / total).item()
  percent_correct = round(accuracy * 100, 2)
  print(f'The model output the correct label {percent_correct}% of the time')
