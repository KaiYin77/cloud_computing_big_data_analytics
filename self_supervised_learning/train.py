'''
Boosted by lightning module
'''
import csv
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from pathlib import Path
import os 
import argparse

'''
Argparse
'''
parser = argparse.ArgumentParser()
parser.add_argument(
        "--train",
        help="train mode",
        action='store_true',
        )
args = parser.parse_args()

class SSRL(pl.LightningModule):
    def __init__(self):
        super().__init__()
    
    def configure_optimizers(self):
        self.warmup_epoch = 3
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=6, gamma=0.75)
        return [self.optimizer], [self.lr_scheduler]

    def optimizer_step(
        self, 
        epoch, 
        batch_idx, 
        optimizer, 
        optimizer_idx, 
        optimizer_closure, 
        on_tpu=False, 
        using_native_amp=False, 
        using_lbfgs=False
    ):
        # update params
        optimizer.step(closure=optimizer_closure)

        # skip the first epochs
        if (self.trainer.global_step < self.warmup_epoch * self.trainer.num_training_batches):
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / float(self.warmup_epoch * self.trainer.num_training_batches))
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.trainer.lr_scheduler_configs[0].scheduler._get_closed_form_lr()[0]

    def training_step(self, train_batch, batch_idx):
        return loss
    
    def training_epoch_end(self, outputs):
        pass

    def validation_step(self, val_batch, val_idx):
        pass 

    def validation_epoch_end(self, outputs):
        pass

    def test_step(self, test_batch, test_idx):
        pass

    def test_epoch_end(self, outputs):
        pass

    def prepare_data(self):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self): 
        pass

if __name__ == '__main__':
    if args.train:
        wandb_logger = WandbLogger(project="self_supervised_learning")
        model = SSRL()
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir, 
            filename=f'{epoch:02d}-{avg_val_loss:.2f}-{val_acc:.2f}',
            save_top_k=5, 
            mode="min",
            monitor="avg_val_loss"
            )
        trainer = pl.Trainer(
            callbacks=[checkpoint_callback],
            accelerator="gpu",
            max_epochs=50,
            logger=wandb_logger,
            gradient_clip_val=1,
            track_grad_norm=2,
            )
        trainer.fit(model)
