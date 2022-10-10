import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torchvision.models import vgg16 

class VGGLSTM(nn.Module):
    def __init__(self, num_class=39):
        super(VGGLSTM, self).__init__()
        self.vgg = vgg16(num_classes=512)
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=3)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_class)

    def forward(self, x_3d):
        hidden = None
        for t in range(x_3d.size(1)):
            x = self.vgg(x_3d[:, t, :, :, :])
            out, hidden = self.lstm(x.unsqueeze(0), hidden)

        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        x = self.fc2(x)
        return x
