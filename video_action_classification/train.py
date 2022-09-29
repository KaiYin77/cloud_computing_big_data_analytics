'''
Boosted by lightning module
'''

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorchvideo.models.resnet
import pytorch_lightning as pl
from dataloader import VideoActionDataset
from pathlib import Path
import os 

train_dir = Path('../data/hw1/train/')
test_dir = Path('../data/hw1/test/')
BATCHSIZE = 1

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(in_dim, out_dim),
                )
    def forward(self, x):
        return self.mlp(x)

class VideoActionClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = self.make_resnet()

    def make_resnet(self):
        return pytorchvideo.models.resnet.create_resnet(
                  input_channel=3,
                  model_depth=50,
                  model_num_class=39,
                  norm=nn.BatchNorm3d,
                  activation=nn.ReLU,
              )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        y_hat = self.model(train_batch["video"])
        loss = F.cross_entropy(y_hat, train_batch["label"])
        return loss
    
    def prepare_data(self):
        self.dataset = VideoActionDataset(train_dir) 
    
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=BATCHSIZE)
    
if __name__ == '__main__':
    model = VideoActionClassifier()
    trainer = pl.Trainer()
    trainer.fit(model)
