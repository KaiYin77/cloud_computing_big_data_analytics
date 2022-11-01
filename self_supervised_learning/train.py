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
import configparser

from dataloader import MRIDataset
from criterion import nt_xent, KNN
from models.baseline import Baseline
from models.resnet import Resnet

''' Argparse
'''
parser = argparse.ArgumentParser()
parser.add_argument(
        "--train",
        help="train mode",
        action='store_true',
        )
parser.add_argument(
        "--val",
        help="val mode",
        action='store_true',
        )
parser.add_argument(
        "--test",
        help="test mode",
        action='store_true',
        )
parser.add_argument(
        "--dev",
        help="dev mode",
        action='store_true',
        )
parser.add_argument(
        "--ckpt", 
        help="specify ckpt name", 
        default="", 
        type=str
        )
args = parser.parse_args()

''' ConfigParser
'''
config =  configparser.ConfigParser()
config.read('config.ini')

''' Constant
'''
_unlabeled_dir = Path(config['data']['unlabeled'])
_test_dir = Path(config['data']['test'])
_ckpt_dir = Path(config['repo']['ckpt_dir'])
_ckpt_name = Path(args.ckpt)
_ckpt_path = _ckpt_dir / _ckpt_name
_submit_dir = Path(config['repo']['submit_dir'])
_submit_path = _submit_dir / Path(f'{_ckpt_name.stem}.npy')

_batch_size = int(config['trainer']['batch_size'])
_save_top_k = int(config['trainer']['save_top_k'])
_max_epochs = int(config['trainer']['max_epochs'])

_warmup_epoch = int(config['optimizer']['warmup_epoch'])
_lr = float(config['optimizer']['lr'])
_weight_decay = float(config['optimizer']['weight_decay'])

_step_size = int(config['scheduler']['step_size'])
_gamma = float(config['scheduler']['gamma'])

class SSRL(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Resnet()
        
        self.train_total = 0
        self.train_loss = 0
    
    def configure_optimizers(self):
        self.warmup_epoch = _warmup_epoch 
        self.optimizer = torch.optim.Adam(self.parameters(), lr=_lr, weight_decay=_weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(self.train_dataloader()), eta_min=0, last_epoch=-1)

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
        x1 = train_batch['image_t1']
        x2 = train_batch['image_t2']

        u_embed, u = self.model(x1)
        v_embed, v = self.model(x2)

        loss = nt_xent(u, v)
        
        self.train_total += 1
        self.train_loss += loss.item()
        self.log('train_loss', self.train_loss/self.train_total, prog_bar=True)
        return loss
    
    def training_epoch_end(self, outputs):
        self.train_total = 0
        self.train_loss = 0

    def validation_step(self, val_batch, val_idx):
        size = val_batch['image_t'].size(0)
        x = val_batch['image_t']
        label = val_batch['label']
        
        embedding, output = self.model(x)
        embedding = embedding.reshape(-1, 512)
        label = label.reshape(-1)
        acc = KNN(embedding, label, batch_size=size)

        return {'acc': acc, 'size': size}

    def validation_epoch_end(self, outputs):
        total_correct = 0
        total_size = 0
        for output in outputs:
            total_correct += output['acc'] * output['size']
            total_size += output['size']
        self.log('val_acc', total_correct/total_size)

    def test_step(self, test_batch, test_idx):
        x = test_batch['image_t']
        embedding, output = self.model(x)
        return embedding

    def test_epoch_end(self, outputs):
        embedding_list = []
        for output in outputs:
            embedding_list.append(output)
        embedding = torch.stack(embedding_list).reshape(-1, 512)

        with open(_submit_path, 'wb') as f:
            np.save(f, embedding.cpu().detach().numpy().astype(np.float32))

    def prepare_data(self):
        self.train_dataset = MRIDataset(_unlabeled_dir, mode="train")
        self.val_dataset = MRIDataset(_test_dir, mode="val")
        self.test_dataset = MRIDataset(_unlabeled_dir, mode="test")

    def train_dataloader(self):
        return DataLoader(
                self.train_dataset,
                batch_size=_batch_size,
                num_workers=8,
                pin_memory=True
                ) 

    def val_dataloader(self):
        return DataLoader(
                self.val_dataset,
                batch_size=_batch_size,
                num_workers=8,
                pin_memory=True
                ) 

    def test_dataloader(self): 
        return DataLoader(
                self.test_dataset,
                batch_size=1,
                num_workers=8,
                pin_memory=True
                ) 

if __name__ == '__main__':
    wandb_logger = WandbLogger(project="self_supervised_learning")
    if args.train:
        model = SSRL()
        checkpoint_callback = ModelCheckpoint(
            dirpath=_ckpt_dir, 
            filename='{epoch:02d}-{train_loss:.2f}-{val_acc:.2f}',
            save_top_k=_save_top_k, 
            mode="max",
            monitor="val_acc"
            )
        trainer = pl.Trainer(
            callbacks=[checkpoint_callback],
            accelerator="gpu",
            max_epochs=_max_epochs,
            logger=wandb_logger,
            gradient_clip_val=1,
            track_grad_norm=2,
            fast_dev_run = True if args.dev else False
            )
        trainer.fit(model)

    if args.val:
        model = SSRL.load_from_checkpoint(
                checkpoint_path=_ckpt_path,
                map_location=None,
                )
        trainer = pl.Trainer(
            accelerator="gpu",
            logger=wandb_logger,
            )
        trainer.validate(model)

    if args.test:
        model = SSRL.load_from_checkpoint(
                checkpoint_path=_ckpt_path,
                map_location=None,
                )
        trainer = pl.Trainer(
            accelerator="gpu",
            logger=wandb_logger,
            )
        trainer.test(model)
