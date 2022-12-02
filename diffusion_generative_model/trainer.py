'''
Boosted by lightning module
'''

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from torchvision import utils
from torchvision.transforms.functional import resize

from PIL import Image

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import os 
import copy
import datetime
from pathlib import Path
import wandb

from parser import create_arg_parser, create_yaml_parser
from dataloader import MNISTDataset, GaussianNoiseDataset
from src.diffusion import DiffusionTrainer, DiffusionSampler
from src.models import UNet
from src.utils import ExponentialMovingAverage

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
ckpt_dir.mkdir(parents=True, exist_ok=True)
ckpt_path = ckpt_dir / Path(str(args.ckpt))

samples_dir = Path(config['repo']['samples_dir'])
samples_dir.mkdir(parents=True, exist_ok=True)

grid_dir = Path(config['repo']['grid_dir'])
grid_dir.mkdir(parents=True, exist_ok=True)

class DDPM(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = UNet(input_shape=(3, 32, 32), **config['model'])
        self.orig_trainer = DiffusionTrainer(
                self.model, time_steps=config['model']['time_steps'])
        self.orig_sampler = DiffusionSampler(
                self.model, time_steps=config['model']['time_steps'])

        self.ema_model = copy.deepcopy(self.model)
        self.ema_sampler = DiffusionSampler(
                self.ema_model, time_steps=config['model']['time_steps'])
        self.ema = ExponentialMovingAverage(self.model,
                                            self.ema_model,
                                            decay=config['trainer']['ema_decay'])
        self.sample_noise = torch.randn(8, 3, 32, 32)
        self.sample_idx = 1
    
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config['optimizer']['lr'])
        return [self.optimizer]

    def training_step(self, batch, idx):
        loss = self.orig_trainer(batch)
        
        self.ema.step()
        self.log('loss/train', loss, prog_bar=True)
        
        return loss
    
    def training_epoch_end(self, outputs):
        if self.current_epoch % config['trainer']['save_freq'] == 0 or self.current_epoch == 1:
            self.orig_sampler.eval()
            self.save_image(self.orig_sampler, prefix='orig')
            self.ema_sampler.eval()
            self.save_image(self.ema_sampler, prefix='ema')
    
    def testing_step(self, batch_x_T, batch_idx):
        batch_x_0 = self.orig_sampler(batch_x_T)
        batch_x_0 = resize(batch_x_0, [28, 28])
        for x_0 in batch_x_0:
            sample_path = samples_dir / Path(f'{self.sample_idx:05d}.png')
            utils.save_image(x_0, sample_path)
            self.sample_idx+=1
        return batch_x_0
    
    @torch.no_grad() 
    def save_image(
            self, 
            sampler: DiffusionSampler,
            num_row: int = 8,
            single_image_size=(28, 28),
            prefix: str = 'sample',
        ) -> None: 
        epoch = self.current_epoch
        sample_x_T = self.sample_noise.to(self.device)

        sample_path =  grid_dir / Path(f'{prefix}_{epoch}.png')
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        
        images = sampler.grid_sample(sample_x_T, num_row)
        images = (images + 1.0) / 2.0  # map [-1, 1] to [0, 1]
        images = resize(images, single_image_size)
        grid = utils.make_grid(images, nrow=num_row)
        
        utils.save_image(grid, sample_path)
        
        upload_image = wandb.Image(grid)
        self.logger.log_image(key="samples", images=[upload_image], caption=[f'{prefix}_{epoch}'])
    
    def prepare_data(self):
        self.train_dataset = MNISTDataset(
                                train_dir, transform=T.Resize((32, 32), InterpolationMode.NEAREST))
        self.test_dataset = GaussianNoiseDataset(
                                (3, 32, 32), config['sampler']['num_samples'])

    def train_dataloader(self):
        return DataLoader(
                self.train_dataset,
                batch_size = config['trainer']['batch_size'],
                num_workers=8,
                pin_memory=True
                )

    def test_dataloader(self):
        return DataLoader(
                self.test_dataset,
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
            filename='{epoch:02d}',
            every_n_epochs=config['trainer']['save_freq'], 
            monitor="epoch"
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
        if args.ckpt:
            trainer.fit(model, ckpt_path=ckpt_path)
        else:
            trainer.fit(model)
    
    if args.test:
        assert args.ckpt != None
        model = DDPM.load_from_checkpoint(
                    checkpoint_path=ckpt_path,
                    map_location=None,
                )
        trainer = pl.Trainer(
                accelerator="gpu",
                logger=wandb_logger,
                )
        trainer.test(model)


