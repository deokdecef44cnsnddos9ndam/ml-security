import IPython.display as display

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import mlsec.utils as ut
from mlsec.imagenet_classes import IMAGENET_CLASSES


def evaluate(model, transform, base_img, desired_class, threshold=0.5):
    """
    Calculates the number of succesfuls classifcations above the target confidence threshold.
    
    Params
    ------
    logits : torch.FloatTensor of shape (1000,)
    desired_class : str
        Targeted class name
    threshold : float
        Confidence value above which defines success
        
    Returns
    -------
    float
        percent success
    """
   
    successes = 0
    failures = 0
    score_history = []
    
    for i in range(5):
        with torch.no_grad():
            probs = model(transform(base_img, 10))
        for l in probs:
            score = ut.get_score(l, desired_class)
            if score > threshold:
                successes += 1
            else:
                failures += 1
                
            score_history.append(score)
            score_history.sort(reverse=True)
            
            percent_success = round((successes / (successes + failures)) * 100.0, 2)
        
            plt.close()
    
            plt.title(f"Percent Success {percent_success}%")
            plt.ylim(0.0, 1.0)
            plt.ylabel('Confidence')
            plt.bar(range(len(score_history)), score_history)
            plt.axhline(0.5, color='red')
    
            display.clear_output(wait=True)
            display.display(plt.gcf())
            
    display.clear_output(wait=True)

def autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def vis_probs(ax, probs, label=None):
    if probs.shape[-1] == 10:
        #mnist
        if label is not None:
            last = ax.bar([label], 1.0, color='red', alpha=0.25)
            ax.legend([last], ['Desired'], loc=1)
        probs = [round(float(p.item()), 2) for p in probs[0]]
        classification_prob = list(map(lambda p: p if p >= 0.5 else 0.0, probs))
        prob_bars = ax.bar(range(10), probs)

        ax.set_title('Model Ouput', pad=20)
        ax.bar(range(10), classification_prob, color='red')
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(range(10))
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylabel('Confidence')
        ax.set_xlabel('Number')

        autolabel(ax, prob_bars)
    else:
        #imagenet
        probs = [float(p.item()) for p in probs[0]]
        labels = list(map(lambda l: l[:26], IMAGENET_CLASSES.values()))
        if label is not None:
            label_score = probs[labels.index(label)]
        data = zip(probs, labels)
        data = sorted(data, key=lambda x: x[0], reverse=True)
        probs, labels = zip(*data[:10])
        labels = list(labels)
        probs = list(probs)
        if label is not None:
            if not (label in labels):
                labels[-1] = label
                probs[-1] = label_score
            last = ax.bar(labels.index(label), 1.0, color='red', alpha=0.25)
            ax.legend([last], ['Desired'], loc=1)
            
        probs = [round(p, 2) for p in probs]
        classification_prob = list(map(lambda p: p if p >= 0.5 else 0.0, probs))
        prob_bars = ax.bar(range(10), probs)
        ax.set_title('Model Ouput', pad=20)
        ax.bar(range(10), classification_prob, color='red')
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(range(10))
        if labels:
            ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylabel('Confidence')
        ax.set_xlabel('Class Name')
        autolabel(ax, prob_bars)
        
def vis_labels(ax, class_id, labels=None):
    if isinstance(class_id, int):
        #mnist
        probs = torch.zeros((1, 10))
        probs[0, class_id] = 1.0
        vis_probs(ax, probs, labels=labels)
        ax.set_title('Desired Ouput', pad=20)
    else:
        #imagenet
        probs = [float(p.item()) for p in probs[0]]
        labels = list(map(lambda l: l[:26], IMAGENET_CLASSES.values()))
        data = zip(probs, labels)
        data = sorted(data, key=lambda x: x[0], reverse=True)
        probs, labels = zip(*data[:10])
        probs = [round(p, 2) for p in probs]
        classification_prob = list(map(lambda p: p if p >= 0.5 else 0.0, probs))
        prob_bars = ax.bar(range(10), probs)
        ax.set_title('Model Ouput', pad=20)
        ax.bar(range(10), classification_prob, color='red')
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(range(10))
        if labels:
            ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylabel('Confidence')
        ax.set_xlabel('Class Name')
        autolabel(ax, prob_bars)
        
def example(img, probs, label=None):
    if probs.shape[-1] == 10:
        #mnist
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
        ut.show_on_axis(ax1, img.repeat(1, 3, 1, 1))
        vis_probs(ax2, probs, label)
    else:
        #imagenet
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
        ut.show_on_axis(ax1, img)
        vis_probs(ax2, probs, label)
        
def example_label(img, probs, label=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
    ut.show_on_axis(ax1, img.repeat(1, 3, 1, 1))
    if label is not None:
        last = ax2.bar([label], 1.0, color='red', alpha=0.25)
        ax2.legend([last], ['Desired'], loc=1)
    probs = [round(float(p.item()), 2) for p in probs[0]]
 
    ax2.bar(range(10), probs, color='red')
    ax2.set_ylim(0.0, 1.0)
    ax2.set_xticks(range(10))
    ax2.set_yticklabels([])
    ax2.set_xlabel('Number')
    
def progress(img, model, loss_history, label):
    probs = model(img).cpu()
    progress_no_inference(img, probs, loss_history, label)
        
def progress_no_inference(img, probs, loss_history, label):
    if probs.shape[-1] == 10:
        #mnist
        plt.close()
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30,5))
        ut.show_on_axis(ax1, img.repeat(1, 3, 1, 1).cpu())

        ax3.set_title('Loss')
        ax3.set_ylabel('Loss Value')
        ax3.set_xlabel('Iteration')
        ax3.plot(loss_history)
        
        vis_probs(ax2, probs, label)
        display.clear_output(wait=True)
        display.display(plt.gcf())
    else:
        #imagenet
        plt.close()
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30,5))

        ut.show_on_axis(ax1, img.cpu())
        vis_probs(ax2, probs, label)
        ax3.set_title('Loss')
        ax3.set_ylabel('Loss Value')
        ax3.set_xlabel('Iteration')
        ax3.plot(loss_history)
        display.clear_output(wait=True)
        display.display(plt.gcf())
