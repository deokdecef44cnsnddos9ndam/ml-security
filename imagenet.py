import pretrainedmodels
import kornia.color as color
import kornia.geometry as geo
import torch
import torch.nn as nn

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
