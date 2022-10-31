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

'''
ConfigParser
'''
config =  configparser.ConfigParser()
config.read('config.ini')

class GaussianBlur():
    def __init__(self, kernel_size, sigma_min=0.1, sigma_max=2.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.kernel_size = kernel_size

    def __call__(self, img):
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        img = cv2.GaussianBlur(np.array(img), (self.kernel_size, self.kernel_size), sigma)
        return Image.fromarray(img.astype(np.uint8))

class MRIDataset(Dataset):
    def __init__(self, raw_dir, mode="train"):
        self.raw_dir = Path(raw_dir)
        self.mode = mode
        self.image_list = sorted(self.raw_dir.rglob("*.jpg"))
        self.size = (96, 96)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = str(self.image_list[idx])
        image = Image.open(image_path) 
        image = np.array(image) 
        
        if self.mode == "train":
            ''' Transform
            '''
            transform = transforms.Compose([
	    	transforms.RandomApply(
	    		[GaussianBlur(kernel_size=23)], 
	    		p=0.1
	    	),
	    	transforms.ToTensor(),
                    transforms.RandomCrop(
	    		self.size, 
	    		padding=16
	    	),
	    	transforms.ColorJitter(
	    		brightness=0.5, 
	    		contrast=0.5, 
	    		saturation=0.5, 
	    		hue=0.5
	    	),
                    transforms.RandomHorizontalFlip(p=0.5)
	    ])
            
            image_t1 = transform(image)
            image_t2 = transform(image)
	
            ''' Stack sample
            '''
            sample = {}
            sample['image_t1'] = image_t1
            sample['image_t2'] = image_t2
        elif self.mode == "test":
            transform = transforms.Compose([
	    	transforms.ToTensor()
	    ])
            image_t = transform(image)
            sample = {}
            sample['image_t'] = image_t
        
        elif self.mode == "val":
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
    batch_size = 4096
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

