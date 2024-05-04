# Build datasets, dataloader
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import numpy as np
import os
import random


def split_dataset(dataset_dir, train_proportion=0.8, val_proportion=0.1):
    # Function that splits a dataset given a path of the folder containing the images
    classes = os.listdir(dataset_dir)
    images = []
    for i in range(len(classes)):
        for file in os.listdir(os.path.join(dataset_dir, classes[i])):
            try:
                images[i].append(
                    os.path.join(dataset_dir, classes[i], file)
                )
            except:
                images.append([os.path.join(dataset_dir, classes[i], file)])

    print("Total number of images: ", sum([len(images[i]) for i in range(len(images))]))
    train_images = [images[i][: int(train_proportion * len(images[i]))] for i in range(len(images))]
    test_images = [images[i][int(train_proportion * len(images[i])) :] for i in range(len(images))]
    
    
    # Further divide the training in validation and training
    val_images = [train_images[i][: int(val_proportion*len(train_images[i]))] for i in range(len(train_images))]
    train_images = [train_images[i][int(val_proportion*len(train_images[i])) :] for i in range(len(train_images))]
    
    return train_images, val_images, test_images

class FlowerDataset(Dataset):
    def __init__(
        self,
        images_list,
        transforms=None,
    ):
        self.transform = transforms
        self.images_list = images_list
        
        self.images_list_flattened = [(i, self.images_list[i][j]) for i in range(len(self.images_list)) for j in range(len(self.images_list[i]))]

    def __len__(self):
        return len(self.images_list_flattened)

    def __getitem__(self, idx):
        class_idx, image_path = self.images_list_flattened[idx]
        # Read image
        image = read_image(image_path, ImageReadMode.RGB) / 255
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, class_idx



def get_train_val_loaders(trainset, valset, batch_size=8):
    
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    valloader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

    return trainloader, valloader

def get_test_loader(testset, batch_size=8):
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )
    
    return testloader


if __name__=="__main__":
    train_images, val_images, test_images = split_dataset("flowers")
    