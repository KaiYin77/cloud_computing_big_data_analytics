
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.models import resnet18
from .vgg16 import VGG16
torch.manual_seed(99)

class ResNetLSTM(pl.LightningModule):
    def __init__(self, num_class=39):
        super(ResNetLSTM, self).__init__()
	self.resnet = resnet18(num_classes=300)
        self.lstm = nn.LSTM(input_size=300, hidden_size=256, num_layers=3)
        self.linear_1 = nn.Linear(2816, 256)
        self.linear_2 = nn.Linear(256, num_class)
        self.dropout = nn.Dropout(p=0.5)

    def init_hidden(self, batch_size):
        return(
          torch.randn(3, batch_size, 256).to(self.device),
          torch.randn(3, batch_size, 256).to(self.device)
        )

    def forward(self, x_3d):
        hidden = self.init_hidden(x_3d.shape[0]) 
        outputs = []
        for t in range(x_3d.shape[1]):
            x = self.resnet(x_3d[:, t, :, :, :])
            output, hidden = self.lstm(x.unsqueeze(0), hidden)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=-1)
        outputs = self.dropout(outputs)
        outputs = self.linear_1(outputs[-1, :, :])
        outputs = self.linear_2(outputs)
        return outputs
