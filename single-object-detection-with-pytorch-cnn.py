from pathlib import Path
import numpy as np
from PIL import Image
from pprint import pprint
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches

torch.manual_seed(0)


import os
import random
import pandas as pd
from pathlib import Path


def load_image_dataset(image_path, label_path, shuffle=False):
    """Returns (image_list, label_list) with full paths."""

    image_path = Path(image_path)
    label_path = Path(label_path)

    # Read the CSV file containing the labels
    labels_df = pd.read_csv(label_path)

    # Get a list of all image filenames (excluding extensions)
    images = [
        str(image_path / img)
        for img in os.listdir(image_path)
        if img.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    # Prepare a dictionary to store labels by image name
    labels_dict = {}
    for _, row in labels_df.iterrows():
        image_name = row["image_name"]
        # Get bounding box coordinates
        bbox = [row["x0"], row["y0"], row["x1"], row["y1"]]
        labels_dict[image_name] = bbox

    # Filter images that have corresponding labels
    images_with_labels = []
    labels = []
    for img in images:
        img_name = os.path.basename(img)
        if img_name in labels_dict:
            images_with_labels.append(img)
            labels.append(labels_dict[img_name])

    # Ensure matching number of images and labels
    if len(images_with_labels) != len(labels):
        raise ValueError("The number of images and labels do not match.")

    if shuffle:
        random.seed(0)
        combined = list(zip(images_with_labels, labels))
        random.shuffle(combined)
        images_with_labels, labels = zip(*combined)

    return list(images_with_labels), list(labels)


def split_dataset(x, y, test_size=0.2, val_size=0.1, random_state=None):
    """Split the dataset into train, validation, and test sets."""
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    # Adjust validation size to the remaining training set
    val_fraction_of_train = val_size / (1 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=val_fraction_of_train,
        random_state=random_state,
    )

    return x_train, x_val, x_test, y_train, y_val, y_test


class ImageDataset(Dataset):
    def __init__(self, x, y, transform=None, target_transform=None):
        super().__init__()
        self.x = x
        self.y = y
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = Image.open(self.x[index]).convert("RGB")
        y = self.y[index]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y


def main(debug=True):

    # Load dataset
    print("Loading dataset...")
    IMAGE_PATH = "dataset/face-detection-data/images"
    LABEL_PATH = "dataset/face-detection-data/faces.csv"
    
    # List images and labels
    images, labels = load_image_dataset(IMAGE_PATH, LABEL_PATH, shuffle=True)
    if debug:
        print(f"    {labels[:3]}")

    # TODO: see images here

    # Split to train and val
    print("Splitting dataset...")
    x_train, x_val, x_test, y_train, y_val, y_test = split_dataset(
        images, labels, test_size=0.2, val_size=0.1, random_state=None
    )
    assert len(x_val) + len(x_test) + len(x_train) == len(images)
    assert len(y_val) + len(y_test) + len(y_train) == len(labels)

    # Create image dataset and transformation
    print("Creating dataset and applying transformation...")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_dataset = ImageDataset(x=x_train, y=y_train, transform=transform)
    val_dataset = ImageDataset(x=x_val, y=y_val, transform=transform)
    test_dataset = ImageDataset(x=x_test, y=y_test, transform=transform)

    # # Create dataloader
    # print("Creating dataloader...")
    # batch_size = 32
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    # if debug:
    #     for batch_idx, (inputs, targets) in enumerate(train_dataloader):
    #         print(f"Batch {batch_idx + 1}")
    #         print(
    #             f"Inputs shape: {inputs.shape}"
    #         )  # Should be (batch_size, channel, H, W)
    #         print(
    #             f"Inputs range: {inputs.max().item(), inputs.min().item()}"
    #         )  # Should be (batch_size, channel, H, W)
    #         print(f"Targets shape: {targets.shape}")  # Should be (batch_size)
    #         print(f"Targets: {targets}")
    #         break


if __name__ == "__main__":
    main(debug=True)
