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
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from dataloader import VideoActionDataset 
from pathlib import Path
import os 
import argparse

from models.vgg_lstm import VGGLSTM
from models.res_lstm import RESLSTM
from models.resnet import RESNET

'''
Argparse
'''
parser = argparse.ArgumentParser()
parser.add_argument(
        "--dev",
        help="dev mode",
        action='store_true',
        )
parser.add_argument(
        "--train",
        help="train mode",
        action='store_true',
        )
parser.add_argument(
        "--test",
        help="test mode",
        action='store_true',
        )
parser.add_argument(
        "--validate",
        help="validate mode",
        action='store_true',
        )
parser.add_argument(
        "--ckpt", 
        help="specify ckpt name", 
        default="", 
        type=str
        )
parser.add_argument(
        "--net", 
        help="specify net name", 
        default="vgglstm", 
        type=str
        )
args = parser.parse_args()

'''
Model Selection (vgglstm/resnetlstm)
'''
NET = args.net

'''
Train Config
'''
train_dir = Path('../data/hw1/train/')
ckpt_dir = Path('./weights/')
BATCHSIZE = 32

'''
Test Config
'''
test_dir = Path('../data/hw1/test/')
test_mini_dir = Path('../data/hw1/test_mini/')
ckpt_name = str(args.ckpt)

class VideoActionClassifier(pl.LightningModule):
    def __init__(self, net="vgglstm"):
        super().__init__()
        if net == "vgglstm":
            self.model = self.make_vgg_lstm()
        elif net == "reslstm":
            self.model = self.make_res_lstm()
        elif net == "resnet":
            self.model = self.make_resnet()
        self.train_total = 0
        self.train_correct = 0
    
    def make_vgg_lstm(self):
        return VGGLSTM(
                num_classes=39
                )

    def make_res_lstm(self):
        return RESLSTM(
                num_classes=39
                )

    def make_resnet(self):
        return RESNET(
                num_classes=39
                )

    def configure_optimizers(self):
        self.warmup_epoch = 3
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=6, gamma=0.75)
        return [self.optimizer], [self.lr_scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # update params
        optimizer.step(closure=optimizer_closure)

        # skip the first epochs
        if (self.trainer.global_step < self.warmup_epoch * self.trainer.num_training_batches):
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / float(self.warmup_epoch * self.trainer.num_training_batches))
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.trainer.lr_scheduler_configs[0].scheduler._get_closed_form_lr()[0]

    def training_step(self, train_batch, batch_idx):
        y_hat = self.model(train_batch["video"])
        loss = F.cross_entropy(y_hat, train_batch["label"])
        self.log("train_loss", loss.item(), prog_bar=True)
        self.log("lr", (self.optimizer).param_groups[0]['lr'])
        conf, index = y_hat.max(-1)
        self.train_total += train_batch['label'].size(0)
        self.train_correct += (index == train_batch['label']).sum().item()
        self.log('train_acc', self.train_correct/self.train_total, prog_bar=True)
        return loss
    
    def training_epoch_end(self, outputs):
        self.train_total = 0
        self.train_correct = 0

    def validation_step(self, val_batch, val_idx):
        y_hat = self.model(val_batch["video"])
        loss = F.cross_entropy(y_hat, val_batch["label"])
        conf, index = y_hat.max(-1)
        val_correct = (index == val_batch['label']).sum().item()
        return {'val_loss': loss.item(), 'val_correct': val_correct}
    
    def validation_epoch_end(self, outputs):
        total_loss = 0
        total_correct = 0
        for output in outputs:
            total_loss += output['val_loss']
            total_correct += output['val_correct']
        self.log('avg_val_loss', total_loss / len(outputs))
        self.log('val_acc', total_correct / len(outputs))

    def test_step(self, test_batch, test_idx):
        y_hat = self.model(test_batch["video"])
        conf, index = y_hat.max(-1)
        return {'video_name': test_batch['video_name'][0], 'predict': index.item()}

    def test_epoch_end(self, outputs):
        submit_file='submit/' + ckpt_name[:-5] + '.csv'
        file = open(submit_file, 'w')
        writer = csv.writer(file)
        data=["name", "label"]
        writer.writerow(data)
        for output in outputs:
            data=[output['video_name'], output['predict']]
            writer.writerow(data)
        file.close()

    def prepare_data(self):
        self.dataset = VideoActionDataset(train_dir, net=NET)
        self.test_dataset = VideoActionDataset(test_dir, mode="test", net=NET)

        val_split = 0.2
        random_seed = 7777
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
                num_workers=8,
                pin_memory=True
                )
    def val_dataloader(self):
        val_sampler = SubsetRandomSampler(self.val_indices)
        return DataLoader(
                self.dataset, 
                batch_size=1, 
                sampler=val_sampler,
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
    if args.train:
        wandb_logger = WandbLogger(project="action_classifier")
        model = VideoActionClassifier(net=NET)
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir, 
            filename=f'{NET}'+'-{epoch:02d}-{avg_val_loss:.2f}-{val_acc:.2f}',
            save_top_k=5, 
            mode="min",
            monitor="avg_val_loss"
            )
        early_stop_callback = EarlyStopping(
            monitor="avg_val_loss",
            mode="min",
            patience=10,
        
        )
        if args.ckpt != "":
            ckpt_path='weights/' + ckpt_name
            trainer = pl.Trainer(
                callbacks=[checkpoint_callback, early_stop_callback],
                accelerator="gpu",
                max_epochs=50,
                logger=wandb_logger,
                gradient_clip_val=1,
                track_grad_norm=2,
                resume_from_checkpoint=ckpt_path
                )
        else:
            trainer = pl.Trainer(
                callbacks=[checkpoint_callback, early_stop_callback],
                accelerator="gpu",
                max_epochs=50,
                logger=wandb_logger,
                gradient_clip_val=1,
                track_grad_norm=2,
                )
        trainer.fit(model)
    if args.validate:
        ckpt_path='weights/' + ckpt_name
        model = VideoActionClassifier.load_from_checkpoint(
                checkpoint_path=ckpt_path,
                net=NET,
                map_location=None,
                )
        trainer = pl.Trainer(
            accelerator="gpu",
            )
        trainer.validate(model)
    if args.test:
        ckpt_path='weights/' + ckpt_name
        model = VideoActionClassifier.load_from_checkpoint(
                checkpoint_path=ckpt_path,
                net=NET,
                map_location=None,
                )
        trainer = pl.Trainer(
            accelerator="gpu",
            )
        trainer.test(model)
    if args.dev:
        model = VideoActionClassifier(net=NET)
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir, 
            filename=f'{NET}'+'-{epoch:02d}-{avg_val_loss:.2f}-{val_acc:.2f}',
            save_top_k=5, 
            mode="min",
            monitor="avg_val_loss"
            )
        trainer = pl.Trainer(
            callbacks=[checkpoint_callback],
            accelerator="gpu",
            max_epochs=50,
            fast_dev_run=True,
            )
        trainer.fit(model)
