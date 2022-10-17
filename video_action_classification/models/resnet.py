import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import resnet18 as resnet
torch.manual_seed(99)
import ipdb

class RESNET(pl.LightningModule):
    def __init__(self, num_classes=39):
        super(RESNET, self).__init__()
        model = resnet(pretrained=False, num_classes=num_classes)
        self.resnet = nn.Sequential(*(list(model.children())[:-1]))
        self.mlp = nn.Sequential(
                  nn.Linear(512*28, 1024),
                  nn.ReLU(),
                  nn.Dropout(p=0.2),
                  nn.Linear(1024, 512),
                  nn.ReLU(),
                  nn.Dropout(p=0.2),
                  nn.Linear(512, 256),
                  nn.ReLU(),
                  nn.Dropout(p=0.1),
                  nn.Linear(256, num_classes),
        )

    def forward(self, x_3d):
        batch_size, timestamp = x_3d.shape[0], x_3d.shape[1]
        
        outputs = []
        for t in range(timestamp):
          output = self.resnet(x_3d[:, t, :, :, :])
          outputs.append(output)
        outputs = torch.stack(outputs)
       	outputs = outputs.reshape(batch_size, -1) 
        outputs = self.mlp(outputs)

        return outputs
