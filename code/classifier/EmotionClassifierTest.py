import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision import datasets, transforms
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from tqdm import tqdm
import os

"""
Facial Emotion Recognition using CNN
CS158 Final Project 2025
Authors: Aiko Kato, Bengisu Bulur, Sophy Figaroa

This script handles data loading and preprocessing for emotion recognition.
"""

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
IMG_SIZE = 48
BATCH_SIZE = 32
SMALL_DATASET = True  # Toggle for testing vs full dataset

# Get the current script directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up two levels to reach project root (from code/classifier to project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
# Dataset is in the 'dataset' folder at project root
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')

# Verify the dataset directory exists
if not os.path.exists(DATASET_DIR):
    print(f"ERROR: Dataset directory not found at {DATASET_DIR}")
else:
    # List the emotion folders found
    emotion_folders = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    print(f"Found emotion folders: {emotion_folders}")

# Adjust for small dataset testing
if SMALL_DATASET:
    BATCH_SIZE = 8  # Smaller batches for limited data

def get_transforms():
    """
    Define image transformations for preprocessing.
    Returns transform for the dataset.
    """
    # Basic preprocessing
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
        transforms.Resize((IMG_SIZE, IMG_SIZE)),      # Resize to 48x48
        transforms.ToTensor(),                         # Convert to tensor [0,1]
        transforms.Normalize(mean=[0.5], std=[0.5])   # Normalize to [-1,1]
    ])
    
    return transform

def load_and_split_dataset(data_dir, transform, test_split=0.2, random_seed=42):
    """
    Load dataset and split into k-fold data and test set.
    
    Args:
        data_dir: Path to data directory containing emotion folders
        transform: Transformations to apply
        test_split: Percentage to hold out for final testing (default 0.2 = 20%)
        random_seed: Random seed for reproducible split
    
    Returns:
        kfold_dataset: Dataset for k-fold cross validation (80%)
        test_dataset: Held-out test dataset (20%)
        class_names: List of emotion class names
    """
    # Check if directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Load the complete dataset
    full_dataset = datasets.ImageFolder(data_dir, transform=transform)
    
    # Get class names
    class_names = full_dataset.classes
    
    print(f"Loaded {len(full_dataset)} total images")
    print(f"Classes: {class_names}")
    
    # Split into k-fold data (80%) and test set (20%)
    total_size = len(full_dataset)
    test_size = int(test_split * total_size)
    kfold_size = total_size - test_size
    
    # Use random_split with a generator for reproducibility
    generator = torch.Generator().manual_seed(random_seed)
    kfold_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [kfold_size, test_size], generator=generator
    )
    
    print(f"Split dataset:")
    print(f"  K-fold data: {len(kfold_dataset)} images (will be split during cross-validation)")
    print(f"  Test set: {len(test_dataset)} images (held-out for final evaluation)")
    
    return kfold_dataset, test_dataset, class_names

def create_test_loader(test_dataset, batch_size=32):
    """
    Create DataLoader for the held-out test set.
    
    Args:
        test_dataset: Test dataset
        batch_size: Batch size for DataLoader
    
    Returns:
        test_loader: DataLoader for test set
    """
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for testing
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    return test_loader

def visualize_batch(dataset, class_names, num_images=8):
    """
    Visualize a batch of images from the dataset.
    
    Args:
        dataset: Dataset to sample from
        class_names: List of class names
        num_images: Number of images to display
    """
    # Create a temporary loader to get a batch
    temp_loader = DataLoader(dataset, batch_size=num_images, shuffle=True)
    images, labels = next(iter(temp_loader))
    
    # Print batch information
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Visualize
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    # Denormalize images for display (from [-1,1] to [0,1])
    images_display = images * 0.5 + 0.5
    
    for i in range(min(num_images, len(images))):
        img = images_display[i].squeeze().cpu()
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"{class_names[labels[i]]}", fontsize=12)
        axes[i].axis('off')
    
    plt.suptitle("Sample Images", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def main():
    """Main function to load data for k-fold cross validation."""
    
    # Get transformations
    transform = get_transforms()
    
    # Load and split dataset
    kfold_dataset, test_dataset, class_names = load_and_split_dataset(
        DATASET_DIR, transform, test_split=0.2
    )
    
    # Create test loader (this stays constant across all k-folds)
    test_loader = create_test_loader(test_dataset, BATCH_SIZE)
    
    print("DATASET SUMMARY")
    print(f"Total images: {len(kfold_dataset) + len(test_dataset)}")
    print(f"K-fold dataset: {len(kfold_dataset)} images")
    print(f"Test dataset: {len(test_dataset)} images")
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}")
    print(f"Test batches: {len(test_loader)}")
    
    # Visualize sample
    visualize_batch(kfold_dataset, class_names)
    
    # Return everything needed for k-fold cross validation
    return {
        'kfold_dataset': kfold_dataset,  # This will be split in k-fold CV
        'test_dataset': test_dataset,     # This is held out
        'test_loader': test_loader,       # Ready-to-use test loader
        'class_names': class_names,
        'num_classes': len(class_names),
        'device': device
    }

# k-fold cross validation
def setup_kfold_loaders(dataset, fold_indices, batch_size=32):
    """
    Create data loaders for a specific fold in k-fold CV.
    
    Args:
        dataset: The k-fold dataset
        fold_indices: Tuple of (train_indices, val_indices) for this fold
        batch_size: Batch size
    
    Returns:
        train_loader, val_loader for this fold
    """
    train_indices, val_indices = fold_indices
    
    # Create samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    # Create loaders
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=val_sampler,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Load the data
    data_dict = main()
    
    # Example: Setting up k-fold cross validation    
    kfold_dataset = data_dict['kfold_dataset']
    test_loader = data_dict['test_loader']
    
    # Setup 10 k-folds
    k_folds = 10  
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # Show fold splits
    for fold, (train_ids, val_ids) in enumerate(kfold.split(range(len(kfold_dataset)))):
        print(f"Fold {fold+1}/{k_folds}: Train={len(train_ids)}, Val={len(val_ids)}")
        
        if fold == 0:  # Just show first fold as example
            train_loader, val_loader = setup_kfold_loaders(
                kfold_dataset, (train_ids, val_ids), BATCH_SIZE
            )
            print(f"  Train batches: {len(train_loader)}")
            print(f"  Val batches: {len(val_loader)}")
    
    print("="*60)