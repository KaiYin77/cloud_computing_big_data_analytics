import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torchvision.models import resnet18 as resnet
import pytorch_lightning as pl

class Resnet(pl.LightningModule):
    def __init__(self):
        super(Resnet, self).__init__()
        model = resnet(pretrained=False)
        self.model = nn.Sequential(*(list(model.children())[:-1]))

    def forward(self, data):
        x = self.model(data)
        x = x.reshape(-1, 512)
        return x
