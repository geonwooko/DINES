import random
import torch
import os
import numpy as np
from loguru import logger
from torchmetrics.functional import f1_score, auroc

def print_params(params):
    """
    print parameters

    Args:
        params: parameter dictionary
    """
    for key, value in params.items():
        logger.info(f'{key} : {value}')


def fix_seed(seed):
    """
    fix seed and make deterministic

    Args:
        seed: random seed
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    
def calculate_metrics(prob, y):
    """
    Calculate the AUC and Macro-F1 metrics

    Args:
        prob: predicted probabilities
        y: ground truth labels

    Returns:
        AUC, Macro_F1
    """
    probs = torch.cat([1-prob, prob], dim=1)
    y_pred = torch.argmax(probs, dim=1)
    AUC = auroc(probs, y, num_classes=2).item()
    Macro_F1 = f1_score(y_pred, y, average="macro", num_classes=2).item()
    return AUC, Macro_F1


def create_directory(directory):
    """
    Create the directory if not exists

    Args:
        directory: directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
