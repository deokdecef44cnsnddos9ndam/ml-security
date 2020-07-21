import pretrainedmodels
import kornia.color as color
import kornia.geometry as geo
import torch
import torch.nn as nn

from mlsec.imagenet_classes import IMAGENET_CLASSES
import mlsec.utils as ut

def build_model(model_name, device):
    return ImagenetModel(model_name).to(device)

class ImagenetModel(nn.Module):
    """
    Different ImageNet models will preprocess images a little differently. The main purpose of the wrapper
    is to define and confirm to a common interface, float tensors in [0, 1]. Preprocessing should be 
    considered part of the models forward pass and totally independant of the data space in which any optimizations 
    will be done.
    """
    
    def __init__(self, model_name: str):
        super(ImagenetModel, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the model
        self.model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained="imagenet")
        self.model = self.model.eval()
        self.model = self.model.to(self.device)
        
        # Explicitly defining the transformation w/ differentiable transforms from kornia
        self.transform = nn.Sequential(
            geo.Resize(self.model.input_size[-2:]),
            color.Normalize(torch.tensor(self.model.mean), torch.tensor(self.model.std)),
        )

        if self.model.input_space == "RGB":
            pass
        elif self.model.input_space == "BGR":
            self.transform = nn.Sequential(transform, color.RgbToBgr())
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, images):
        """
        Model's forward pass.
        
        Params
        ------
        images : torch.FloatTensor in [0, 1] with shape [B, C, H, W]
        
        Returns
        -------
        torch.FloatTensor
            logits of shape [B, 1000]
        """
        images = images.to(self.device)
        transformed_images = self.transform(images)
        logits = self.model(transformed_images)
        return self.softmax(logits)


def get_inference(logits):
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

def get_class_index(class_name):
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
        
def print_inference(logits):
    """
    Prints the top 5 class and their confidences
    
    Params
    ------
    logits : torch.FloatTensor of shape (1000,)
    """
    results = get_inference(logits)
    for name, prob in results:
        print(f'{name}: {prob}')
        
def get_score(logits, class_name):
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
    probs = nn.Softmax(dim=0)(logits)
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
