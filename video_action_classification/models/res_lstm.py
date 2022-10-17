import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import resnet34 as resnet
torch.manual_seed(99)
import ipdb

class RESLSTM(pl.LightningModule):
    def __init__(self, num_classes=39):
        super(RESLSTM, self).__init__()
        model = resnet(pretrained=False, num_classes=num_classes)
        self.resnet = nn.Sequential(*(list(model.children())[:-1]))
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=1)
        self.mlp = nn.Sequential(
                  nn.Linear(256*28, 256),
                  nn.ReLU(),
                  nn.Dropout(p=0.2),
                  nn.Linear(256, 256),
                  nn.ReLU(),
                  nn.Dropout(p=0.2),
                  nn.Linear(256, 256),
                  nn.ReLU(),
                  nn.Dropout(p=0.1),
                  nn.Linear(256, num_classes),
        )

    def init_hidden(self, batch_size):
        return(
          torch.randn(1, batch_size, 256).to(self.device),
          torch.randn(1, batch_size, 256).to(self.device)
        )

    def forward(self, x_3d):
        batch_size, timestamp = x_3d.shape[0], x_3d.shape[1]
        
        outputs = []
        for t in range(timestamp):
          output = self.resnet(x_3d[:, t, :, :, :])
          outputs.append(output)
        outputs = torch.stack(outputs)
        
        outputs = outputs.reshape(timestamp, batch_size, -1) 
        hidden = self.init_hidden(batch_size) 
        outputs, hidden = self.lstm(outputs, hidden)
        
        outputs = outputs.permute(1, 0, 2) 
       	outputs = outputs.reshape(batch_size, -1) 
        outputs = self.mlp(outputs)

        return outputs
