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
from dataloader import VideoActionDataset, VideoActionTestDataset
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
parser.add_argument(
        "--test",
        help="test mode",
        action='store_true',
        )
args = parser.parse_args()
'''
Config
'''
train_dir = Path('../data/hw1/train/')
processed_dir = Path('../data/hw1/processed/')
test_dir = Path('../data/hw1/test/')
test_processed_dir = Path('../data/hw1/test_processed/')
ckpt_dir = Path('./weights/')
BATCHSIZE = 8

os.makedirs(processed_dir, exist_ok=True)
os.makedirs(test_processed_dir, exist_ok=True)

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

    def test_step(self, test_batch, test_idx):
        y_hat = self.model(test_batch["video"])
        conf, index = y_hat.max(-1)
        return {'video_name': test_batch['video_name'], 'predict': index}

    def test_epoch_end(self, outputs):
        file = open('./submit/311511036.csv', 'w')
        writer = csv.writer(file)
        data=["name", "label"]
        writer.writerow(data)
        for output in outputs:
            pass
        file.close()

    def prepare_data(self):
        self.dataset = VideoActionDataset(train_dir, processed_dir)
        self.test_dataset = VideoActionTestDataset(test_dir, test_processed_dir)

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
        model = VideoActionClassifier()
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir, 
            save_top_k=5, 
            monitor="val_loss"
            )
        trainer = pl.Trainer(
            callbacks=[checkpoint_callback],
            accelerator="gpu",
            max_epochs=50,
            )
        trainer.fit(model)
    if args.test:
        model = VideoActionClassifier.load_from_checkpoint(
                checkpoint_path="./weights/epoch=0-step=3000.ckpt",
                map_location=None,
                )
        trainer = pl.Trainer(
            accelerator="gpu",
            )
        trainer.test(model)
