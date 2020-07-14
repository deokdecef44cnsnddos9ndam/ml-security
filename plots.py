import IPython.display as display
import matplotlib.pyplot as plt
from mlsec.imagenet_classes import IMAGENET_CLASSES

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
    
def plot_progress(loss_history, logits_history, base_class, target_class):
    plt.close()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
    
    ax1.set_title("Loss")
    ax1.plot(loss_history)
    ax1.set_xlabel("Iteration")
    
    ax2.set_title("Classes")
    ax2.set_ylim(0.0, 1.0)
    
    top = sum([get_inference(lh[0]) for lh in logits_history], [])
    classes = list(map(lambda l: l[0], top))
    classes = set(classes) - set([base_class, target_class])
    classes = [base_class, target_class] + list(classes)
    
    for c in classes[:20]:
        scores = list(map(lambda l: get_score(l[0], c), logits_history))
        ax2.plot(scores)
        
    ax2.legend(list(map(lambda l: l[:50], classes)), bbox_to_anchor=(1.04,1), loc="upper left")
    ax2.set_ylabel('Confidence')
    ax2.set_xlabel('Iteration')
    display.clear_output(wait=True)
    display.display(fig)
