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
            frame = self.crop_center_square(frame)
            frame = cv2.resize(frame, (self.height, self.width))
            frame_list[frame_count] = torch.from_numpy(frame)
            frame_count += 1

        sample = {}
        sample['video'] = frame_list
        parent_path = os.path.dirname(self.all_video_list[idx])
        parent_dir = os.path.split(parent_path)[1]
        sample['label'] = int(parent_dir)
        return sample

    def crop_center_square(self, frame):
        y, x = frame.shape[0:2]
        min_dim = min(y, x)
        start_x = (x // 2) - (min_dim // 2)
        start_y = (y // 2) - (min_dim // 2)
        return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]

if __name__ == '__main__':
    '''
    Unit test
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
            print('label: ', data['label'])
        assert data['video'].shape == torch.Size([batch_size, 82, 90, 90, 3]), 'Frame shape should be (batch_size, 82, 90, 90, 3)'
