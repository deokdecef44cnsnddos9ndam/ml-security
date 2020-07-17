import torch
import torchvision

import matplotlib.pyplot as plt
from skimage import io
import numpy as np

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
    grid = torchvision.utils.make_grid(images, nrow=nrow)
    show(grid.unsqueeze(0), figsize=figsize)   
    
def show_transform_examples(transform, img, n_examples=4):
    batched_img = img.repeat(n_examples, 1, 1, 1)
    image_grid(transform(batched_img), nrow=2)
