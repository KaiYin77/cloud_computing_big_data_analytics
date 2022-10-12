import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import vgg16
torch.manual_seed(99)
import ipdb

class VGGLSTM(pl.LightningModule):
    def __init__(self, num_class=39):
        super(VGGLSTM, self).__init__()
        self.vgg = vgg16(pretrained=False).features[:28]
        self.lstm = nn.LSTM(input_size=25088, hidden_size=30, num_layers=1)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.linear_1 = nn.Linear(30, 90)
        self.linear_2 = nn.Linear(90, 512)
        self.linear_3 = nn.Linear(512, num_class)
        self.dropout_1 = nn.Dropout(p=0.1)
        self.dropout_2 = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()

    def init_hidden(self, batch_size):
        return(
          torch.randn(1, batch_size, 30).to(self.device),
          torch.randn(1, batch_size, 30).to(self.device)
        )

    def forward(self, x_3d):
        hidden = self.init_hidden(x_3d.shape[0]) 
        outputs = []
        for t in range(x_3d.shape[1]):
            x = self.vgg(x_3d[:, t, :, :, :])
            x = x.reshape(1, x_3d.shape[0], -1)
            output, hidden = self.lstm(x, hidden)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=-1)
        outputs = self.linear_1(outputs[-1].permute(0, 2, 1))
        outputs = self.dropout_1(outputs)
        outputs = self.pooling(outputs.permute(0, 2, 1))
        
        outputs = self.linear_2(outputs.reshape(x_3d.shape[0], -1))
        outputs = self.relu(outputs)
        outputs = self.dropout_2(outputs)
        outputs = self.linear_3(outputs)
        return outputs
