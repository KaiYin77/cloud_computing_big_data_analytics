'''
Boosted by lightning module
'''

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl

class VideoActionClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        self.log('val_loss: ', loss)
    
    def prepare_data(self):
        self.train_dataset = 
        self.val_dataset = 
    
    def train_dataloader(self):
        return DataLoader()
    
    def val_dataloader(self):
        return DataLoader()

if __name__ == '__main__':
    model = VideoActionClassifier()
    trainer = pl.Trainer()
    trainer.fit(model)
