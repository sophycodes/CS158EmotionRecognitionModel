"""
Test Model Evaluation
Evaluate trained emotion recognition models on the test dataset

Usage:
    python test_model.py --model_path results/large_cnn_TIMESTAMP/model.pth --model_type large_cnn
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import numpy as np
import argparse
import os
from datetime import datetime

# Import model architectures
from models import BasicNN, SimpleCNN, MediumCNN, LargeCNN

# Configuration
img_size = 48
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def get_model(model_name, num_classes):
    """Create model based on name"""
    if model_name == 'basic_nn':
        model = BasicNN(num_classes)
    elif model_name == 'simple_cnn':
        model = SimpleCNN(num_classes)
    elif model_name == 'medium_cnn':
        model = MediumCNN(num_classes)
    elif model_name == 'large_cnn':
        model = LargeCNN(num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model.to(device)


def load_test_data(test_path, batch_size=64):
    """
    Load test dataset with same preprocessing as training
    """
    # CRITICAL: Same preprocessing pipeline as training
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Load test dataset
    test_dataset = datasets.ImageFolder(test_path, transform=transform)
    class_names = test_dataset.classes
    num_classes = len(class_names)
    
    print(f"Loaded {len(test_dataset)} test images")
    print(f"Classes: {class_names}")
    
    # Create test loader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_loader, class_names, num_classes


def evaluate_model(model, test_loader, class_names):
    """
    Evaluate model on test set and return metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    print("\nEvaluating model on test set...")
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            # Track predictions
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Track accuracy
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    # Calculate overall accuracy
    accuracy = correct / total
    print(f"\nTest Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    # Calculate confusion matrix
    labels = list(range(len(class_names)))
    cm = confusion_matrix(all_labels, all_preds, labels=labels)
    
    # Generate classification report
    report = classification_report(
        all_labels, all_preds,
        labels=labels,
        target_names=class_names,
        zero_division=0
    )
    
    print("\nClassification Report:")
    print(report)
    
    return accuracy, cm, report, all_labels, all_preds


def plot_confusion_matrix(cm, class_names, save_path):
    """
    Plot and save confusion matrix
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=class_names,
                yticklabels=class_names,
                cmap='Blues')
    plt.title('Test Set Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: {save_path}")
    plt.close()


def plot_per_class_accuracy(cm, class_names, save_path):
    """
    Plot per-class accuracy bar chart
    """
    # Calculate per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, per_class_acc, color='skyblue', edgecolor='navy')
    plt.xlabel('Emotion Class')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy on Test Set')
    plt.ylim([0, 1])
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Per-class accuracy plot saved to: {save_path}")
    plt.close()


def save_results(accuracy, cm, report, class_names, save_dir):
    """
    Save all evaluation results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save metrics as text file
    with open(f"{save_dir}/test_results.txt", 'w') as f:
        f.write(f"Test Set Evaluation Results\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        f.write(f"Classification Report:\n")
        f.write(report)
        f.write(f"\n\nConfusion Matrix:\n")
        f.write(f"Classes: {class_names}\n")
        f.write(np.array2string(cm))
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, class_names, f"{save_dir}/test_confusion_matrix.png")
    
    # Plot per-class accuracy
    plot_per_class_accuracy(cm, class_names, f"{save_dir}/test_per_class_accuracy.png")
    
    print(f"\nAll results saved to: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Test emotion recognition model')
    
    # Model selection
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model weights (.pth file)')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['basic_nn', 'simple_cnn', 'medium_cnn', 'large_cnn'],
                       help='Model architecture type')
    
    # Test data path
    parser.add_argument('--test_path', type=str, 
                       default='/cs158/TestData/images/test',
                       help='Path to test dataset directory')
    
    # Batch size
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for testing (default: 64)')
    
    # Output directory
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save results (default: test_results_TIMESTAMP)')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    # Check if test data exists
    if not os.path.exists(args.test_path):
        raise FileNotFoundError(f"Test data directory not found: {args.test_path}")
    
    print(f"\n{'='*60}")
    print(f"Testing {args.model_type} model")
    print(f"Model weights: {args.model_path}")
    print(f"Test data: {args.test_path}")
    print(f"{'='*60}\n")
    
    # Load test data
    test_loader, class_names, num_classes = load_test_data(
        args.test_path, args.batch_size
    )
    
    # Create model
    model = get_model(args.model_type, num_classes)
    
    # Load trained weights
    print(f"Loading model weights from: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print("Model loaded successfully!")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")
    
    # Evaluate model
    accuracy, cm, report, y_true, y_pred = evaluate_model(
        model, test_loader, class_names
    )
    
    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"test_results_{args.model_type}_{timestamp}"
    
    # Save results
    save_results(accuracy, cm, report, class_names, args.output_dir)
    
    print(f"\n{'='*60}")
    print(f"Testing Complete!")
    print(f"Final Test Accuracy: {accuracy:.4f}")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()