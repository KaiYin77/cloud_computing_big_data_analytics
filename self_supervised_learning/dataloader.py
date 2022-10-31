import numpy as np
import pandas as pd
import cv2
import PIL.Image as Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import sys
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from os import listdir
from os.path import isfile, join

import argparse
import configparser
from tqdm import tqdm
from pathlib import Path

from utils import GaussianBlur

'''
ConfigParser
'''
config =  configparser.ConfigParser()
config.read('config.ini')

class MRIDataset(Dataset):
    def __init__(self, raw_dir, mode="train"):
        self.raw_dir = Path(raw_dir)
        self.mode = mode
        self.image_list = sorted(self.raw_dir.rglob("*.jpg"))
        self.size = 96

    def __len__(self):
        return len(self.image_list)

    def get_simclr_pipeline_transform(self, size, s=1):
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size=int(0.1 * size)),
            transforms.ToTensor()
        ])
        return data_transforms  

    def __getitem__(self, idx):
        image_path = str(self.image_list[idx])
        image = Image.open(image_path) 
        
        if self.mode == "train":
            ''' Transform
            '''
            transform = self.get_simclr_pipeline_transform(self.size) 
            image_t1 = transform(image)
            image_t2 = transform(image)
	
            ''' Stack sample
            '''
            sample = {}
            sample['image_t1'] = image_t1
            sample['image_t2'] = image_t2
        elif self.mode == "test":
            image = np.array(image) 
            transform = transforms.Compose([
	    	transforms.ToTensor()
	    ])
            image_t = transform(image)
            sample = {}
            sample['image_t'] = image_t
        
        elif self.mode == "val":
            image = np.array(image) 
            transform = transforms.Compose([
	    	transforms.ToTensor()
	    ])
            image_t = transform(image)
            sample = {}
            sample['image_t'] = image_t
            
            parent_path = os.path.dirname(self.image_list[idx])
            parent_dir = os.path.split(parent_path)[1]
            sample['label'] = torch.as_tensor(int(parent_dir))

        return sample

if __name__ == '__main__':
    '''
    Unit test
    '''
    batch_size = 512
    unlabeled_dir = Path('../data/hw2/unlabeled/')
    dataset = MRIDataset(unlabeled_dir, mode="train") 
    dataloader = DataLoader(dataset, batch_size=batch_size)
    dataiter = tqdm(dataloader)
    for i, data in enumerate(dataiter):
        x1 = data['image_t1']
        x2 = data['image_t2']
        assert x1.shape == torch.Size([batch_size, 3, 96, 96]), 'image shape should be (batch_size, 3, 96, 96)'
        assert x2.shape == torch.Size([batch_size, 3, 96, 96]), 'image shape should be (batch_size, 3, 96, 96)'
        exit()

