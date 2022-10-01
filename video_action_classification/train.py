'''
Boosted by lightning module
'''
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import pytorchvideo.models.resnet
import pytorch_lightning as pl
from dataloader import VideoActionDataset
from pathlib import Path
import os 

train_dir = Path('../data/hw1/train/')
processed_dir = Path('../data/hw1/processed/')
test_dir = Path('../data/hw1/test/')
ckpt_dir = Path('./weights/')
BATCHSIZE = 8

os.makedirs(processed_dir, exist_ok=True)

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
                  head_pool_kernel_size=(4,3,3),
              )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-1)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        y_hat = self.model(train_batch["video"])
        loss = F.cross_entropy(y_hat, train_batch["label"])
        self.log("train_loss", loss.item())
        return loss
    
    def validation_step(self, val_batch, val_idx):
        y_hat = self.model(val_batch["video"])
        loss = F.cross_entropy(y_hat, val_batch["label"])
        self.log("val_loss", loss.item())

    def prepare_data(self):
        self.dataset = VideoActionDataset(train_dir, processed_dir)
        
        val_split = 0.2
        random_seed = 1234
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))

        split = int(np.floor(val_split * dataset_size))
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        self.train_indices, self.val_indices = indices[split:], indices[:split]

    def train_dataloader(self):
        train_sampler = SubsetRandomSampler(self.train_indices)
        return DataLoader(
                self.dataset, 
                batch_size=BATCHSIZE, 
                sampler=train_sampler,
                num_workers=8
                )
    def val_dataloader(self):
        val_sampler = SubsetRandomSampler(self.val_indices)
        return DataLoader(
                self.dataset, 
                batch_size=1, 
                sampler=val_sampler,
                num_workers=8
                )
    
if __name__ == '__main__':
    model = VideoActionClassifier()
    trainer = pl.Trainer(default_root_dir=ckpt_dir)
    trainer.fit(model)
