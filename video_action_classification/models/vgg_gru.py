
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
import pytorch_lightning as pl
from .vgg16 import VGG16
torch.manual_seed(99)

class GRUNet(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:,-1]))
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden

class VGGGRU(pl.LightningModule):
    def __init__(self, num_class=39):
        super(VGGLSTM, self).__init__()
        self.vgg = VGG16(num_classes=300)
        self.lstm = GRUNET(input_size=300, hidden_size=256, num_layers=3)
        self.linear_1 = nn.Linear(2816, 256)
        self.linear_2 = nn.Linear(256, num_class)
        self.dropout = n.Dropout(p=0.5)

    def forward(self, x_3d):
        hidden = self.init_hidden(x_3d.shape[0]) 
        outputs = []
        for t in range(x_3d.shape[1]):
            x = self.vgg(x_3d[:, t, :, :, :])
            output, hidden = self.lstm(x.unsqueeze(0), hidden)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=-1)
        outputs = self.dropout(outputs)
        outputs = self.linear_1(outputs[-1, :, :])
        outputs = self.linear_2(outputs)
        return outputs
