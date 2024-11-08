import os
import random
from pathlib import Path
from pprint import pprint

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split

torch.manual_seed(0)


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


def plot_random_sample(images, bboxes, mean=None, std=None):
    # Randomly select an index to plot
    index = random.randint(0, len(images) - 1)
    image_path = images[index]
    bbox = bboxes[index]  # [x0, y0, x1, y1]

    # Open and possibly denormalize the image
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)

    # If the image was normalized, denormalize it for display
    if mean is not None and std is not None:
        image = (image / 255.0 - mean) / std * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)

    # Create a plot
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image)

    # Draw the bounding box on the image
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min

    # Create and add the bounding box rectangle
    rect = patches.Rectangle(
        (x_min, y_min),
        width,
        height,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
    )
    ax.add_patch(rect)
    ax.set_title(f"Sample Image with Bounding Box")
    plt.show()


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

    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def _transform(self, x, target_size):
        """Do transformation to images."""
        transformer = transforms.Compose(
            [
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        return transformer(x)

    def _target_transform(self, y, original_size, target_size):
        """
        Resize bounding box coordinates according to new image dimensions.
        """
        x0, y0, x1, y1 = y
        original_width, original_height = original_size
        target_width, target_height = target_size

        # Calculate scaling factors
        scale_x = target_width / original_width
        scale_y = target_height / original_height

        # Resize bounding box
        new_x0 = x0 * scale_x
        new_y0 = y0 * scale_y
        new_x1 = x1 * scale_x
        new_y1 = y1 * scale_y

        return torch.tensor([new_x0, new_y0, new_x1, new_y1])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        target_size = (128, 128)
        x = Image.open(self.x[index]).convert("RGB")
        original_size = x.size
        x = self._transform(x, target_size)
        y = self.y[index]
        y = self._target_transform(y, original_size, target_size)
        return x, y


def train_one_step(model, x, y, optimizer, loss_fn):
    optimizer.zero_grad()
    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def train_one_epoch(model, input, optimizer, loss_fn):
    model.train()
    running_loss = 0.0
    for batch_idx, (x, y) in enumerate(input):
        loss = train_one_step(model, x, y, optimizer, loss_fn)
        running_loss += loss
    return running_loss / (batch_idx + 1)


def val_one_step(model, x, y, loss_fn):
    output = model(x)
    loss = loss_fn(output, y)
    return loss.item()


def val_one_epoch(model, input, loss_fn):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(input):
            loss = val_one_step(model, x, y, loss_fn)
            running_loss += loss
    return running_loss / (batch_idx + 1)


class CNN(nn.Module):
    def __init__(
        self,
    ):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=6,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=12,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=12288, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def plot_predictions(images, true_labels, pred_labels):
    """
    Plots images with their true and predicted bounding boxes.

    Parameters:
        images (list): List of images (torch tensors) from the test set.
        true_labels (list): List of true bounding box coordinates.
        pred_labels (list): List of predicted bounding box coordinates.
    """
    num_samples = len(images)

    for i in range(num_samples):
        image = (
            images[i][0].permute(1, 2, 0).cpu().numpy()
        )  # Remove batch dim and reshape
        image = (
            image * 0.5 + 0.5
        ) * 255  # Denormalize if normalized with mean 0.5, std 0.5
        image = image.astype(np.uint8)

        # Get the true and predicted bounding box coordinates
        true_bbox = true_labels[i][0].cpu().numpy()
        pred_bbox = pred_labels[i][0].cpu().numpy()

        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(image)

        # Plot true bounding box in green
        true_x0, true_y0, true_x1, true_y1 = true_bbox
        true_width = true_x1 - true_x0
        true_height = true_y1 - true_y0
        rect_true = patches.Rectangle(
            (true_x0, true_y0),
            true_width,
            true_height,
            linewidth=2,
            edgecolor="green",
            facecolor="none",
        )
        ax.add_patch(rect_true)
        ax.text(true_x0, true_y0 - 5, "True", color="green", fontsize=12, weight="bold")

        # Plot predicted bounding box in red
        pred_x0, pred_y0, pred_x1, pred_y1 = pred_bbox
        pred_width = pred_x1 - pred_x0
        pred_height = pred_y1 - pred_y0
        rect_pred = patches.Rectangle(
            (pred_x0, pred_y0),
            pred_width,
            pred_height,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect_pred)
        ax.text(
            pred_x0, pred_y0 - 5, "Predicted", color="red", fontsize=12, weight="bold"
        )

        plt.axis("off")
        plt.show()


def main(debug=True):

    # Load dataset
    print("Loading dataset...")
    IMAGE_PATH = "dataset/face-detection-data/images"
    LABEL_PATH = "dataset/face-detection-data/faces.csv"

    # List images and labels
    images, labels = load_image_dataset(IMAGE_PATH, LABEL_PATH, shuffle=True)
    if debug:
        print(f"    {labels[:3]}")

    # See sample images
    if debug:
        for i in range(5):
            plot_random_sample(images, labels, mean=None, std=None)

    # Split to train and val
    print("Splitting dataset...")
    x_train, x_val, x_test, y_train, y_val, y_test = split_dataset(
        images, labels, test_size=0.2, val_size=0.1, random_state=None
    )
    assert len(x_val) + len(x_test) + len(x_train) == len(images)
    assert len(y_val) + len(y_test) + len(y_train) == len(labels)

    # Create image dataset and transformation
    print("Creating dataset and applying transformation...")
    train_dataset = ImageDataset(x=x_train, y=y_train)
    val_dataset = ImageDataset(x=x_val, y=y_val)
    test_dataset = ImageDataset(x=x_test, y=y_test)

    # Create dataloader
    print("Creating dataloader...")
    batch_size = 4
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    if debug:
        for _, (inputs, targets) in enumerate(train_dataloader):
            # Should be (batch_size, channel, H, W)
            print(f"    Inputs shape: {inputs.shape}")
            print(f"    Inputs range: {inputs.max().item(), inputs.min().item()}")
            # Should be (batch_size)
            print(f"    Targets shape: {targets.shape}")
            print(f"    Targets: {targets}")
            break

    # Create model
    print("Creating model...")
    if debug:
        samples = torch.zeros(size=(4, 3, 128, 128))
        sample_model = CNN()
        sample_output_size = sample_model(samples).shape
        print(f"    Sample output size = {sample_output_size}")
    model = CNN()

    # Train and validate
    print("Training...")
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    epochs = 10
    print_step = 1
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        # training
        train_loss = train_one_epoch(model, train_dataloader, optimizer, loss_fn)
        train_losses.append(train_loss)
        # validation
        val_loss = val_one_epoch(model, val_dataloader, loss_fn)
        val_losses.append(val_loss)
        if (epoch == 0) or ((epoch + 1) % print_step == 0):
            # print train and validation result
            avg_train_loss = np.mean(train_loss)
            avg_val_loss = np.mean(val_loss)
            print(
                f"  Epoch {epoch+1: <3}/{epochs} | train MSE = {avg_train_loss: .8f} | val MSE = {avg_val_loss: .8f}"
            )

    # Test and see test result
    print("Testing...")
    images = []
    pred_labels = []
    true_labels = []
    for i, (x, y) in enumerate(test_dataloader):
        if i == 10:
            break
        model.eval()
        with torch.no_grad():
            output = model(x)
            images.append(x)
            pred_labels.append(output)
            true_labels.append(y)
    plot_predictions(images, true_labels, pred_labels)


if __name__ == "__main__":
    main(debug=True)
