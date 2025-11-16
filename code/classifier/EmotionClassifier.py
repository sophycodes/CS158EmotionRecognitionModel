import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from tqdm import tqdm
import os

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data paths
base_dir = 'path/to/facial-emotion-recognition-dataset'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Image transformations
img_size = 48
batch_size = 32

# Apply simple transformations to ensure consistent input
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
    transforms.Resize((48, 48)),                   # Ensure size 48x48
    transforms.ToTensor(),                         # Convert to tensor [0,1]
    transforms.Normalize(mean=[0.5], std=[0.5])    # Normalize to [-1,1] keeps numbers small and centered around zero to make traininf faster
])

# Do the same transformations for test
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# loading, labeling, batching, and shuffling images
# Load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
# 1. Scan all subfolders in data
# 2. Each subfolder becomes a class (angry=0, happy=1, sad=2, etc.)
# 3. Loads all images from each subfolder
# 4. Applies the transform to each image
# 5. Creates (image, label) pairs

test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

# FOR INITIAL TESTING: Using only 10 images per emotion
# 7 emotions × 10 images = 70 total training images
# (Will scale up to full dataset on HPC later)

# Split training data for validation (80-20 split)
train_size = int(0.8 * len(train_dataset))  # 80% for training (56 images)
val_size = len(train_dataset) - train_size  # 20% for validation (14 images)
train_dataset, val_dataset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size]
)

# Create data loaders
train_loader = DataLoader(
    train_dataset,        # The dataset to load from
    batch_size=32,        # Load 32 images at a time 
    shuffle=True          # Randomize order each epoch (important for training!)
                         # Prevents model from memorizing order
                         # Each epoch sees different batch combinations
)

# Validation loader - for checking model performance DURING training
val_loader = DataLoader(
    val_dataset,          # 14 validation images (2 per emotion on average)
    batch_size=batch_size, # Process 32 at a time (will only have 1 batch of 14)
    shuffle=False         # IMPORTANT: No shuffling for validation!
                         # - Ensures consistent evaluation across epochs
                         # - Makes validation scores comparable epoch-to-epoch
                         # - Allows reproducible results
                         # - Order doesn't matter since we're not training on these
)

# Test loader - for final evaluation AFTER training is complete
test_loader = DataLoader(
    test_dataset,         # Completely separate test images
    batch_size=32,        # Process in batches of 32
    shuffle=False         # Keep test order consistent (for reproducibility)
                         # Same order every run = same final score
)

# Print dataset info for verification
print("="*50)
print("DATASET SIZES (Initial Small Version):")
print(f"Training images: {len(train_dataset)} ({len(train_dataset)//7} per emotion)")
print(f"Validation images: {len(val_dataset)} ({len(val_dataset)//7} per emotion)")
print(f"Test images: {len(test_dataset)}")
print(f"Number of classes: {len(train_dataset.dataset.classes)}")
print(f"Classes: {train_dataset.dataset.classes}")
print("="*50)
print(f"Batches per epoch - Train: {len(train_loader)}")
print(f"Batches per epoch - Val: {len(val_loader)}")
print(f"Batches per epoch - Test: {len(test_loader)}")
print("="*50)
print("NOTE: This is a small subset for initial testing.")
print("Full dataset will be used when running on HPC.")


# Get class names
class_names = train_dataset.dataset.classes
num_classes = len(class_names)
print(f"Classes: {class_names}")
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")



def check_data_loaded_correctly(data_loader):
    """Function to make sure images are loading properly"""
    images, labels = next(iter(data_loader))
    
    print(f"Batch shape: {images.shape}")  # Should be [32, 1, 48, 48]
    print(f"Labels shape: {labels.shape}")  # Should be [32]
    
    # Show a few images
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    axes = axes.ravel()
    
    # Denormalize for display
    images_display = images * 0.5 + 0.5
    
    emotion_names = train_dataset.classes
    
    for i in range(8):
        img = images_display[i].squeeze()
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"{emotion_names[labels[i]]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

check_data_loaded_correctly(train_loader)