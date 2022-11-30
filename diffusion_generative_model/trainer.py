'''
Boosted by lightning module
'''

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import os 
from pathlib import Path

from parser import create_arg_parser, create_yaml_parser
from dataloader import MNISTDataset

''' Argparse
'''
args = create_arg_parser()

''' ConfigParser
'''
config = create_yaml_parser()

''' Constant
'''
root = Path(config['data']['root'])
train_dir = root / Path('mnist') 
npz_path = root / Path('mnist.npz')

ckpt_dir = Path(config['repo']['ckpt_dir'])
ckpt_path = ckpt_dir / Path(str(args.ckpt))

class DDPM(pl.LightningModule):
    def __init__(self):
        super().__init__()
    
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config['optimizer']['lr'])
        return [self.optimizer]

    def training_step(self, batch):
        pass

    def prepare_data(self):
        self.train_dataset = MNISTDataset(train_dir)

    def train_dataloader(self):
        return DataLoader(
                self.train_dataset,
                batch_size = config['trainer']['batch_size'],
                num_workers=8,
                pin_memory=True
                )

if __name__ == '__main__':
    wandb_logger = WandbLogger(project="denoising diffusion probabilistic model")
    if args.train:
        model = DDPM()
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir, 
            filename='{epoch:02d}-{train_loss:.2f}-{val_acc:.2f}',
            save_top_k=config['trainer']['save_top_k'], 
            mode="max",
            monitor="val_acc"
            )
        trainer = pl.Trainer(
            callbacks=[checkpoint_callback],
            accelerator="gpu",
            max_epochs=config['trainer']['max_epochs'],
            logger=wandb_logger,
            gradient_clip_val=1,
            track_grad_norm=2,
            fast_dev_run = True if args.dev else False
            )
        trainer.fit(model)


