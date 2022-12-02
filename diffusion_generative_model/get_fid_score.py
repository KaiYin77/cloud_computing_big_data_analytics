import os
from tqdm import tqdm

import torch
from torchvision.io import read_image

from pytorch_gan_metrics import get_fid

images = []
for i in tqdm(range(1, 10001)):
    path = os.path.join(f'./samples/{i:05d}.png')
    image = read_image(path) / 255.
    images.append(image)
images = torch.stack(images, dim=0)
FID = get_fid(images, '../data/lab2/mnist.npz')
print(f'FID: {FID:.5f}')

