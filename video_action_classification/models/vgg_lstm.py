import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from .vgg16 import VGG16

class VGGLSTM(nn.Module):
    def __init__(self, num_class=39):
        super(VGGLSTM, self).__init__()
        self.vgg = VGG16(num_classes=300)
        self.lstm = nn.LSTM(input_size=300, hidden_size=256, num_layers=3)
        self.linear = nn.Linear(256, num_class)

    def forward(self, x_3d):
        hidden = None
        for t in range(x_3d.size(1)):
            x = self.vgg(x_3d[:, t, :, :, :])
            out, (final_hidden_state, final_cell_state) = self.lstm(x.unsqueeze(0), hidden)

        x = self.linear(out[-1, :, :])
        return x
