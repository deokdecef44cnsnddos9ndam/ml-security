import torch
import torch.nn as nn
import torchvision

import matplotlib.pyplot as plt
from skimage import io
import numpy as np
from typing import Sequence, Tuple

from mlsec.imagenet_classes import IMAGENET_CLASSES

# psuedo-types for illustration
Image = Images = torch.FloatTensor
Logit = Logits = torch.FloatTensor
Gradient = Gradients = torch.FloatTensor

def load_image_as_tensor(image_path: str) -> Image:
    """
    Loads an image from image_path into a tensor

    Parameters
    ----------
    image_path: str
        Path to an image

    Returns
    -------
        Image
    """
    img = io.imread(image_path)
    if len(img.shape) == 2:
        img = np.tile(np.expand_dims(img, -1), [1, 1, 3])
    img = torch.FloatTensor(img)
    img = img.permute(2, 0, 1)
    img = img.unsqueeze(0)
    return img / 255.0

def show(img: Image, figsize=None, label=None):
    """
    Displays the image in the notebook window
    
    Params
    ------
    img : Image
    """
    fig, ax = plt.subplots(figsize=figsize)
    show_on_axis(ax, img, label)
  
def show_on_axis(ax, img: Image, label=None):
    """
    Displays the image on specific axis
    
    Params
    ------
    img : Image
    """
    img = img.detach()
    img = img[0].numpy()
    img = img.transpose(1, 2, 0)
    ax.imshow(img, interpolation='nearest', aspect='equal')
    ax.set_xticks([], minor=[])
    ax.set_yticks([], minor=[])
    if label:
        ax.set_xlabel(label)
    
def image_grid(images, nrow, figsize):
    grid = torchvision.utils.make_grid(images, nrow=nrow, pad_value=1)
    show(grid.unsqueeze(0), figsize=figsize)   
    
def show_transform_examples(transform, img, n_examples=4):
    batched_img = img.repeat(n_examples, 1, 1, 1)
    image_grid(transform(batched_img), nrow=2)

def get_inference(logits: Logits) -> Sequence[Tuple[str, float]]:
    """
    Returns the top five highest confidence classes and their probabilities.
    
    Params
    ------
    logits : torch.FloatTensor of shape (1000,)
    
    Returns
    -------
    List[Tuple[str, float]]
        ordered list of top 5 classes and their probability
    """
    probs = nn.Softmax(dim=0)(logits)
    values, indeces = probs.topk(5)
    results = []
    for v, idx in zip(values, indeces):
        class_id = idx.item()
        class_name = IMAGENET_CLASSES[class_id]
        results += [(class_name, v)]
    return results

def get_class_index(class_name: str) -> int:
    """
    Returns the class index for a given class name.
    
    Params
    ------
    class_name: str
        Name of imagenet class
    
    Returns
    -------
    int
        class index
    """
    ind, _ = next(filter(lambda x: x[1] == class_name, IMAGENET_CLASSES.items()), None)
    return ind
        
def print_inference(logits: Logits):
    """
    Prints the top 5 class and their confidences
    
    Params
    ------
    logits : torch.FloatTensor of shape (1000,)
    """
    results = get_inference(logits)
    for name, prob in results:
        print(f'{name}: {prob}')
        
def get_score(probs, class_name: str) -> float:
    """
    Returns the probability of a given class
    
    Params
    ------
    logits : torch.FloatTensor of shape (1000,)
    class_name: str
        name of class
    
    Returns
    -------
    float
        class probability
    """
    ind = get_class_index(class_name)
    return probs[ind].item()

def make_labels(class_name: str, size: int) -> torch.LongTensor:    
    """
    Make a set of standard torch labels for a given class. For use with nn.CrossEntropyLoss
    
    Params
    ------
    class_name: str
        name of class
    size: int
        number of labels
    
    Returns
    -------
    torch.LongTensor of shape (size,)
        labels
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_index = get_class_index(class_name)
    labels = class_index * torch.ones((size)).long()
    return labels.to(device)
