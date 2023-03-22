# Build datasets, dataloader
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import numpy as np
import os
import random


def _init_fn(worker_id):
    np.random.seed(0 + worker_id)


# Create the dataset
class cats_dogs_dataset(Dataset):
    def __init__(self, image_dir, image_list, transforms=None, train=True):
        self.images_dir = image_dir
        self.transforms = transforms
        self.images_list = image_list
        self.train = train

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        # Read the image
        path = os.path.join(self.images_dir, self.images_list[idx])
        img = read_image(path, mode=ImageReadMode.RGB) / 255
        if self.transforms != None:
            # Apply transforms
            img = self.transforms(img)
        # Get the label
        if self.train:
            label = self.images_list[idx].split(".")[0]
            if label == "cat":
                label = torch.tensor(0.0)
            elif label == "dog":
                label = torch.tensor(1.0)
            else:
                raise (RuntimeError("Label not recognized!"))
            return img, label

        return img


def get_datasets(
    train_images_dir,
    test_images_dir,
    validation_split=0,
    train_percentage=1,
    transforms_train=None,
    transforms_val_test=None,
):
    all_training_imgs = os.listdir(train_images_dir)
    random.shuffle(all_training_imgs)
    all_training_imgs = all_training_imgs[
        : int(train_percentage * len(all_training_imgs))
    ]  # DISCARD SOME IMAGES TO SPEED UP TRAINING

    val_imgs = all_training_imgs[: int(validation_split * len(all_training_imgs))]
    train_imgs = all_training_imgs[int(validation_split * len(all_training_imgs)) :]
    test_imgs = os.listdir(test_images_dir)

    trainset = cats_dogs_dataset(
        train_images_dir, train_imgs, transforms=transforms_train, train=True
    )
    valset = cats_dogs_dataset(
        train_images_dir, val_imgs, transforms=transforms_val_test, train=True
    )
    testset = cats_dogs_dataset(
        test_images_dir, test_imgs, transforms=transforms_val_test, train=False
    )

    return trainset, valset, testset


def get_loaders(trainset, valset, testset, batch_size, num_workers=0, pin_memory=True):
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_init_fn,
        persistent_workers=True,
    )
    valloader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        worker_init_fn=_init_fn,
    )
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        worker_init_fn=_init_fn,
    )

    return trainloader, valloader, testloader
