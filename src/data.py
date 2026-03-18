from typing import List, Tuple

import numpy as np
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch


class BeansTorchDataset(Dataset):
    def __init__(self, hf_split, image_transform=None):
        self.data = hf_split
        self.transform = image_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        image = item["image"]
        label = item["labels"]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))

        image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(image_size: int, batch_size: int, num_workers: int):
    ds = load_dataset("beans")
    label_names = ds["train"].features["labels"].names
    num_classes = len(label_names)

    train_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = BeansTorchDataset(ds["train"], train_tfms)
    val_dataset = BeansTorchDataset(ds["validation"], eval_tfms)
    test_dataset = BeansTorchDataset(ds["test"], eval_tfms)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, label_names, num_classes