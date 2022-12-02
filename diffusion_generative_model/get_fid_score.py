from pytorch_gan_metrics import get_fid
from torchvision.io import read_image

images = []
for i in range(num_images):
    path = os.path.join(f'./samples/{i:05d}.png')
    image = read_image(path) / 255.
    images.append(image)
images = torch.stack(images, dim=0)
FID = get_fid(images, '../data/lab2/mnist.npz')
print(f'FID: {FID:.5f}')

