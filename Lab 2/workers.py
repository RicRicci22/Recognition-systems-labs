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


class FlowerDataset(Dataset):
    def __init__(
        self,
        root_dir,
        train_proportion=0.8,
        val_proportion=0.1,
        split="train",
        transform=None,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.split = split
        self.train_proportion = train_proportion
        self.val_proportion = val_proportion
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.images = []
        self.labels = []
        for i in range(len(self.classes)):
            for file in os.listdir(os.path.join(root_dir, self.classes[i])):
                try:
                    self.images[self.class_to_idx[self.classes[i]]].append(
                        os.path.join(root_dir, self.classes[i], file)
                    )
                except:
                    self.images.insert(
                        self.class_to_idx[self.classes[i]],
                        [os.path.join(root_dir, self.classes[i], file)],
                    )

        # In questo modo ho creato una lista di lunghezza num classes, dove ogni elemento è una lista di tutte le path delle immagini di quella classe
        train_images = [
            int(self.train_proportion * len(self.images[i]))
            for i in range(len(self.classes))
        ]
        val_images = [
            int(self.val_proportion * train_images[i]) for i in range(len(self.classes))
        ]

        self.train_images = [
            class_images[: train_images[i]]
            for i, class_images in enumerate(self.images)
        ]
        self.test_images = [
            class_images[train_images[i] :]
            for i, class_images in enumerate(self.images)
        ]
        self.val_images = [
            class_images[: val_images[i]]
            for i, class_images in enumerate(self.train_images)
        ]
        self.train_images = [
            class_images[val_images[i] :]
            for i, class_images in enumerate(self.train_images)
        ]

        # Reshape val and test images to not pick randomly

        self.val_images_flattened = [
            (i, self.val_images[i][j])
            for i in range(len(self.val_images))
            for j in range(len(self.val_images[i]))
        ]

        self.test_images_flattened = [
            (i, self.test_images[i][j])
            for i in range(len(self.test_images))
            for j in range(len(self.test_images[i]))
        ]

    def __len__(self):
        if self.split == "train":
            return sum([len(self.train_images[i]) for i in range(len(self.classes))])
        elif self.split == "val":
            return len(self.val_images_flattened)
        else:
            return len(self.test_images_flattened)

    def __getitem__(self, idx):
        # Extract randomly a class to sample from
        class_idx = random.randint(0, len(self.classes) - 1)
        if self.split == "train":
            image_path = random.choice(self.train_images[class_idx])
            image = read_image(image_path, ImageReadMode.RGB) / 255
            if self.transform:
                image = self.transform(image)
            return image, class_idx
        else:
            return self.getitem_deterministic(idx)

    def getitem_deterministic(self, idx):
        if self.split == "test":
            image_path = self.test_images_flattened[idx][1]
            label = self.test_images_flattened[idx][0]
        elif self.split == "val":
            image_path = self.val_images_flattened[idx][1]
            label = self.val_images_flattened[idx][0]
        image = read_image(image_path, ImageReadMode.RGB) / 255
        if self.transform:
            image = self.transform(image)
        # Just return the image and label, transformation applied in main
        return image, label


def get_loaders(transforms_train, transforms_test_val, batch_size=8):
    trainset = FlowerDataset("flowers", split="train", transform=transforms_train)
    valset = FlowerDataset("flowers", split="val", transform=transforms_test_val)
    testset = FlowerDataset("flowers", split="test", transform=None)

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        worker_init_fn=_init_fn,
        pin_memory=True,
        persistent_workers=True,
    )
    valloader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=_init_fn,
        pin_memory=True,
    )

    return trainloader, valloader, testset
