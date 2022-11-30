from typing_extensions import Literal

import torch
from torch import nn
from torch.nn import functional as F

from src.models import UNet

if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
