import numpy as np
import pandas as pd
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from os import listdir
from os.path import isfile, join
import argparse
from tqdm import tqdm
from pathlib import Path
import sys

class VideoActionDataset(Dataset):
    def __init__(self, raw_dir, mode="trainval", net="vgglstm"):
        self.raw_dir = raw_dir
        self.mode = mode
        self.net = net
        self.all_video_list = sorted(self.raw_dir.rglob("*.mp4"))
        self.max_len = 82
        self.height = 112
        self.width = 112
        self.channel = 3
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

    def __len__(self):
        return len(self.all_video_list)

    def __getitem__(self, idx):
        video_path = str(self.all_video_list[idx])
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            success = True
        else:
            sucess = False
            print('loading .mp4 error...')

        frame_list = torch.zeros(self.max_len, self.channel, self.height, self.width)
        frame_count = 0
        while(success):
            success, frame = cap.read()
            if success is False:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.height, self.width), interpolation=cv2.INTER_AREA)

            frame = self.transform(frame)
            '''
            Test
            '''
            #from PIL import Image
            #if frame_count % 8 == 0:
            #    print('frame_count: ', frame_count)
            #    pil_image=Image.fromarray(frame)
            #    pil_image.save(f'temp/{idx}_{frame_count}.jpeg')
            '''
            Append
            '''
            frame_list[frame_count] = frame
            frame_count += 1

        
        '''
        Stack all sample
        '''
        sample = {}
        frame_list = frame_list[::3] # downsample to 1hz frame rate
        #if self.net == "vgglstm":
        frame_list = torch.permute(frame_list, (0, 1, 2, 3))
        sample['video'] = frame_list
        if self.mode == "test":
          sample['video_name'] = os.path.basename(video_path)
        else:
          parent_path = os.path.dirname(self.all_video_list[idx])
          parent_dir = os.path.split(parent_path)[1]
          sample['label'] = torch.as_tensor(int(parent_dir))

        return sample

if __name__ == '__main__':
    '''
    Unit test: train/val dataset
    '''
    debug = False 
    batch_size = 2
    train_dir = Path('../data/hw1/train/')
    dataset = VideoActionDataset(train_dir) 
    dataloader = DataLoader(dataset, batch_size=batch_size)
    dataiter = tqdm(dataloader)
    for i, data in enumerate(dataiter):
        if debug:
            print('video.shape: ', data['video'].shape)
            print('label: ', data['label'].shape)
        #(B, C, T, H, W)
        assert data['video'].shape == torch.Size([batch_size, 11, 3, 90, 90]), 'Video shape should be (batch_size, 11, 3, 90, 90)'
        #(B)
        assert data['label'].shape == torch.Size([batch_size]), 'Label shape should be (batch_size)'
