import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import vgg16
torch.manual_seed(99)
import ipdb

class VGGLSTM(pl.LightningModule):
    def __init__(self, num_classes=39):
        super(VGGLSTM, self).__init__()
        self.vgg = vgg16(pretrained=False, num_classes=num_classes, dropout=0).features[:28]
        self.lstm = nn.LSTM(input_size=25088, hidden_size=30, num_layers=1)
        self.mlp = nn.Sequential(
                  nn.Linear(840, 128),
                  nn.ReLU(),
                  nn.Linear(128, 128),
                  nn.ReLU(),
                  nn.Linear(128, 128),
                  nn.ReLU(),
                  nn.Linear(128, num_classes),
        )

    def init_hidden(self, batch_size):
        return(
          torch.randn(1, batch_size, 30).to(self.device),
          torch.randn(1, batch_size, 30).to(self.device)
        )

    def forward(self, x_3d):
        batch_size = x_3d.shape[0]
        hidden = self.init_hidden(batch_size) 
        outputs = []
        for t in range(x_3d.shape[1]):
            x = self.vgg(x_3d[:, t, :, :, :])
            x = x.reshape(1, x_3d.shape[0], -1)
            x, hidden = self.lstm(x, hidden)
            outputs.append(x)

        outputs = torch.stack(outputs, dim=-1)
       	outputs = outputs.reshape(batch_size, -1) 
        outputs = self.mlp(outputs)

        return outputs
