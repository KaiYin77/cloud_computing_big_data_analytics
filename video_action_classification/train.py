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
import pytorchvideo.models.resnet
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from dataloader import VideoActionDataset 
from pathlib import Path
import os 
import argparse

from model import VGGLSTM

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
        default="epoch=11", 
        type=str
        )
parser.add_argument(
        "--net", 
        help="specify net name", 
        default="resnet", 
        type=str
        )
args = parser.parse_args()

'''
Model Selection (resnet/vgglstm)
'''
NET = args.net

'''
Train Config
'''
train_dir = Path('../data/hw1/train/')
ckpt_dir = Path('./weights/')
BATCHSIZE = 16

'''
Test Config
'''
test_dir = Path('../data/hw1/test/')
test_mini_dir = Path('../data/hw1/test_mini/')
ckpt_name = str(args.ckpt)

class VideoActionClassifier(pl.LightningModule):
    def __init__(self, net="resnet"):
        super().__init__()
        if net == "resnet": 
            self.model = self.make_resnet()
        elif net == "vgglstm":
            self.model = self.make_vgg_lstm()
        self.train_total = 0
        self.train_correct = 0
    
    def make_vgg_lstm(self):
        return VGGLSTM(
                num_class=39
                )

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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        #optimizer = torch.optim.SGD(self.parameters(), lr=1e-2, weight_decay=1e-3, momentum=0.9)
        #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100,150])
        #return [optimizer], [lr_scheduler]
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        y_hat = self.model(train_batch["video"])
        loss = F.cross_entropy(y_hat, train_batch["label"])
        self.log("train_loss", loss.item(), prog_bar=True)
        
        conf, index = y_hat.max(-1)
        self.train_total += train_batch['label'].size(0)
        self.train_correct += (index == train_batch['label']).sum().item()
        self.log('train_acc', self.train_correct/self.train_total, prog_bar=True)
        return loss
    
    def validation_step(self, val_batch, val_idx):
        y_hat = self.model(val_batch["video"])
        loss = F.cross_entropy(y_hat, val_batch["label"])

        conf, index = y_hat.max(-1)
        val_correct = (index == val_batch['label']).sum().item()
        return {'val_loss': loss, 'val_correct': val_correct}
    
    def validation_epoch_end(self, outputs):
        total_loss = 0
        total_correct = 0
        for output in outputs:
            total_loss += output['val_loss'].item()
            total_correct += output['val_correct']
        self.log('avg_val_loss', total_loss / len(outputs))
        self.log('val_acc', total_correct / len(outputs))

    def test_step(self, test_batch, test_idx):
        y_hat = self.model(test_batch["video"])
        conf, index = y_hat.max(-1)
        return {'video_name': test_batch['video_name'][0], 'predict': index.item()}

    def test_epoch_end(self, outputs):
        submit_file='submit/' + ckpt_name + '.csv'
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

        val_split = 0.15
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
            mode="max",
            monitor="val_acc"
            )
        trainer = pl.Trainer(
            callbacks=[checkpoint_callback],
            accelerator="gpu",
            max_epochs=50,
            logger=wandb_logger,
            )
        trainer.fit(model)
    if args.validate:
        ckpt_path='weights/' + ckpt_name + '.ckpt'
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
        ckpt_path='weights/' + ckpt_name + '.ckpt'
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
            mode="max",
            monitor="val_acc"
            )
        trainer = pl.Trainer(
            callbacks=[checkpoint_callback],
            accelerator="gpu",
            max_epochs=50,
            fast_dev_run=True,
            )
        trainer.fit(model)
