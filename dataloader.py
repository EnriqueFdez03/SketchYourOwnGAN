import os
from os import listdir
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.io import read_image
from torch.utils import data
import torch.nn.functional as F

import torch
import torch.nn as nn
from PIL import Image
from networks.usePhotoSketch import PhotoSketchTrain

from diffaug import DiffAugment

# destinado a la carga de imágenes, ya sean bocetos o imágenes
class ImageDataset(Dataset):
    def __init__(self,path,transform=None, image_mode = "RGB") -> None:
        self.files = [os.path.join(path,file) for file in os.listdir(path)]
        self.transform = transform
        self.image_mode = image_mode

    def __getitem__(self, idx):
        imagefile = self.files[idx]
        image = Image.open(imagefile).convert(self.image_mode)   
        if self.transform is not None:
            image = self.transform(image)
        
        return image
    
    def __len__(self):
        return len(self.files)

def create_dataloader(dir, size, batch, img_channel=3):
    mean, std = [0.5 for _ in range(img_channel)], [0.5 for _ in range(img_channel)]
    transform = transforms.Compose([
        transforms.Resize((size,size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std, inplace=True)
    ])
    
    dataset = ImageDataset(dir,transform, "RGB" if img_channel==3 else "L")
    sampler = data.RandomSampler(dataset)
    return data.DataLoader(dataset, batch_size=batch, sampler=sampler, drop_last=True)


# Si disponemos de 50 imagenes y un tamaño de batch de 4, podríamos
# como mucho recorrer 50/4 batches, a partir de ahí stopiteration error.
# Yield_data permite samplear tanto como se deseeel dataset dado un tam
# de batch.
def yield_data(loader):
    while True:
        for batch in loader:
            yield batch

# Transform con el objetivo de: dada imagen obtener boceto y dado boceto
# obtener 3 canales.
class SketchTransforms(nn.Module):
    def __init__(self, toSketch=False, augment=False):
        super(SketchTransforms, self).__init__()
        self.toSketch = toSketch
        self.diffaug = DiffAugment()
        
        transforms = []
        if(self.toSketch):
            transforms.append(PhotoSketchTrain("networks/photosketch.pth"))
        transforms.append(ThreeChannel())
        if augment:
            transforms.append(self.diffaug)

        self.transforms = nn.Sequential(*transforms)

    def forward(self, img):
        img = self.transforms(img)
        return img


class ThreeChannel(nn.Module):
    def __init__(self):
        super(ThreeChannel, self).__init__()
      
    def forward(self, img):
        return img.repeat(1, 3, 1, 1)