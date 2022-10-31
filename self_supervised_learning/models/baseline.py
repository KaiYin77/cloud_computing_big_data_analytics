import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
import pytorch_lightning as pl

class Baseline(pl.LightningModule):
    def __init__(self):
        super(Baseline, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=5),
            nn.AvgPool2d(1),
            nn.Flatten(start_dim=1),
        )

    def forward(self, data):
        x = self.model(data)
        return x
