import numpy as np
import pandas as pd
import cv2

import torch
from torch.utils.data import Dataset, DataLoader

import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from os import listdir
from os.path import isfile, join
import argparse
from tqdm import tqdm
from pathlib import Path
import sys

class VideoActionDataset(Dataset):
    def __init__(self, raw_dir):
        self.raw_dir = raw_dir
        self.all_video_list = sorted(self.raw_dir.rglob("*.mp4"))
        self.max_len = 82
        self.height = 90
        self.width = 90
        self.channel = 3

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

        frame_list = torch.zeros(self.max_len, self.height, self.width, self.channel)
        frame_count = 0
        while(success):
            success, frame = cap.read()
            if success is False:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.height, self.width), interpolation=cv2.INTER_AREA)
            
            '''
            Test
            '''
            #from PIL import Image
            #if frame_count % 4 == 0:
            #    print('frame_count: ', frame_count)
            #    pil_image=Image.fromarray(frame)
            #    pil_image.save(f'temp/{idx}_{frame_count}.jpeg')
            '''
            Perform linear transformation to 0~1
            '''
            frame_list[frame_count] = torch.from_numpy(frame)/255.0
            frame_count += 1

        
        '''
        Stack all sample
        '''
        sample = {}
        frame_list = frame_list[::4] # downsample to 1/4 frame rate
        sample['video'] = torch.permute(frame_list, (3, 0, 1, 2))
        parent_path = os.path.dirname(self.all_video_list[idx])
        parent_dir = os.path.split(parent_path)[1]
        sample['label'] = torch.as_tensor(int(parent_dir))

        return sample

    def crop_center_square(self, frame):
        y, x = frame.shape[0:2]
        min_dim = min(y, x)
        start_x = (x // 2) - (min_dim // 2)
        start_y = (y // 2) - (min_dim // 2)
        return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]

class VideoActionTestDataset(Dataset):
    def __init__(self, raw_dir):
        self.raw_dir = raw_dir
        self.all_video_list = sorted(self.raw_dir.rglob("*.mp4"))
        self.max_len = 82
        self.height = 90
        self.width = 90
        self.channel = 3

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

        frame_list = torch.zeros(self.max_len, self.height, self.width, self.channel)
        frame_count = 0
        while(success):
            success, frame = cap.read()
            if success is False:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.height, self.width), interpolation=cv2.INTER_AREA)
            frame_list[frame_count] = torch.from_numpy(frame)/255.0
            frame_count += 1

        sample = {}
        frame_list = frame_list[::4] # downsample to 1/4 frame rate
        sample['video'] = torch.permute(frame_list, (3, 0, 1, 2))
        sample['video_name'] = os.path.basename(video_path)

        return sample

    def crop_center_square(self, frame):
        y, x = frame.shape[0:2]
        min_dim = min(y, x)
        start_x = (x // 2) - (min_dim // 2)
        start_y = (y // 2) - (min_dim // 2)
        return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]

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
        assert data['video'].shape == torch.Size([batch_size, 3, 21, 90, 90]), 'Video shape should be (batch_size, 3, 82, 90, 90)'
        #(B)
        assert data['label'].shape == torch.Size([batch_size]), 'Label shape should be (batch_size)'
