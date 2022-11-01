import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torchvision.models import resnet18 as resnet
import pytorch_lightning as pl

class Resnet(pl.LightningModule):
    def __init__(self):
        super(Resnet, self).__init__()
        self.backbone = resnet(pretrained=False, num_classes=512)
        self.final_layer = nn.Sequential(
                nn.Linear(512, 256), 
                nn.ReLU(), 
                nn.Linear(256, 128), 
        )

    def forward(self, data):
        embedding = self.backbone(data)
        output = self.final_layer(embedding)
        return embedding, output
