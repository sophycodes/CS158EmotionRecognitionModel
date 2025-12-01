"""
SmallDataTrainerV2.py
Train emotion recognition models with command-line arguments

Loss Function: cross-entropy loss
Algorithm: ADAM (gradient descent w/two moving averages for each weight Average of recent gradients (momentum) + Average of recent squared gradients)
Usage:
    python SmallDataTrainerV2.py --model simple_cnn --epochs 20 --batch_size 32 --lr 0.001
    python SmallDataTrainerV2.py --model feedforward --epochs 30
    python SmallDataTrainerV2.py --model medium_cnn --epochs 50 --batch_size 64
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
import os
import json
import argparse
from datetime import datetime

# Import all model architectures from models.py
from models import BasicNN, SimpleCNN, MediumCNN

# Configuration (can be overridden by command-line args)
img_size = 48
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset loading
def get_data_loaders(batch_size, dataset_path='../../dataset'):
    """Load and prepare data loaders"""
    
    # Preprocessing pipeline
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Load dataset
    # automatically creates class names from dataset folder structure.
    full_dataset = datasets.ImageFolder(dataset_path, transform=transform)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    
    print(f"Loaded {len(full_dataset)} total images")
    print(f"Classes: {class_names}")
    
    # 80-20 train-val split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create data loaders: feeds data to the model in small, manageable batches and feeds to model one batch at a time during training.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, class_names, num_classes


def get_model(model_name, num_classes):
    """Create model based on name"""

    if model_name == 'basic_nn':
        model = BasicNN(num_classes)
    elif model_name == 'simple_cnn':
        model = SimpleCNN(num_classes)
    elif model_name == 'medium_cnn':
        model = MediumCNN(num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: basic_nn, simple_cnn, medium_cnn")

    return model.to(device)


def train_one_epoch(model, train_loader, criterion, optimizer, epoch, total_epochs):
    """
    Train for one epoch so the model has seen every single training image exactly once
    
    """
    
    model.train()  # Set to training mode
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    print(f"Epoch {epoch+1}/{total_epochs} - Loss: {avg_loss:.4f}, Train Acc: {accuracy:.4f}")
    
    return avg_loss, accuracy


def evaluate(model, val_loader, criterion, class_names):
    """Evaluate on validation set"""
    
    model.eval()  # Set to evaluation mode
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    
    print(f"Validation - Loss: {avg_loss:.4f}, Val Acc: {accuracy:.4f}")
    
    # Calculate per-class metrics
    # FIX: Specify all possible labels (0-6 for 7 classes)
    labels = list(range(len(class_names)))
    
    report = classification_report(all_labels, all_preds, 
                                   labels=labels,  # ADDED
                                   target_names=class_names,
                                   output_dict=True,
                                   zero_division=0)
    
    cm = confusion_matrix(all_labels, all_preds, labels=labels)  # ADDED labels here too
    
    return avg_loss, accuracy, report, cm, all_labels, all_preds


def save_results(args, train_history, val_history, final_report, cm, class_names):
    """Save all results to disk"""
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"results/{args.model}_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    
    # Save model weights
    model_path = f"{result_dir}/model.pth"
    torch.save(args.trained_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Save metrics as JSON
    metrics = {
        'model_name': args.model,
        'hyperparameters': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
        },
        'final_val_accuracy': val_history['accuracy'][-1],
        'final_train_accuracy': train_history['accuracy'][-1],
        'per_class_metrics': final_report,
        'confusion_matrix': cm.tolist(),
        'training_history': {
            'train_loss': train_history['loss'],
            'train_accuracy': train_history['accuracy'],
            'val_loss': val_history['loss'],
            'val_accuracy': val_history['accuracy']
        }
    }
    
    with open(f"{result_dir}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Plot and save learning curves
    plot_learning_curves(train_history, val_history, result_dir)
    
    # Plot and save confusion matrix
    plot_confusion_matrix(cm, class_names, result_dir)
    
    print(f"Results saved to {result_dir}")
    return result_dir


def plot_learning_curves(train_history, val_history, save_dir):
    """Plot and save learning curves"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(train_history['loss'], label='Train Loss')
    ax1.plot(val_history['loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(train_history['accuracy'], label='Train Acc')
    ax2.plot(val_history['accuracy'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/learning_curves.png", dpi=150)
    plt.close()


def plot_confusion_matrix(cm, class_names, save_dir):
    """Plot and save confusion matrix"""
    
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=class_names,
                yticklabels=class_names,
                cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix.png", dpi=150)
    plt.close()


def train(args):
    """Main training function"""
    
    print(f"\n{'='*50}")
    print(f"Training {args.model}")
    print(f"{'='*50}\n")
    
    # Load data
    train_loader, val_loader, class_names, num_classes = get_data_loaders(
        args.batch_size, args.dataset_path
    )
    
    # Create model
    model = get_model(args.model, num_classes)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training history
    train_history = {'loss': [], 'accuracy': []}
    val_history = {'loss': [], 'accuracy': []}
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch, args.epochs
        )
        train_history['loss'].append(train_loss)
        train_history['accuracy'].append(train_acc)
        
        # Validate
        val_loss, val_acc, report, cm, y_true, y_pred = evaluate(
            model, val_loader, criterion, class_names
        )
        val_history['loss'].append(val_loss)
        val_history['accuracy'].append(val_acc)
        
        # Save best model, since later epochs might have overfitting
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
        
        print()  # Empty line between epochs
    
    print(f"\n{'='*50}")
    print(f"Training Complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"{'='*50}\n")
    
    # Final evaluation
    final_val_loss, final_val_acc, final_report, final_cm, y_true, y_pred = evaluate(
        model, val_loader, criterion, class_names
    )
    
    # Print classification report
    print("\nPer-Class Performance:")
    labels = list(range(len(class_names)))
    print(classification_report(y_true, y_pred, labels=labels, target_names=class_names, zero_division=0))
    
    # Save results
    args.trained_model = model  # Store for saving
    result_dir = save_results(args, train_history, val_history, final_report, final_cm, class_names)
    
    return model, result_dir


def main():
    parser = argparse.ArgumentParser(description='Train emotion recognition models')
    
    # Model selection
    parser.add_argument('--model', type=str, required=True,
                       choices=['feedforward', 'simple_cnn', 'medium_cnn'],
                       help='Model architecture to train')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    
    # Data path
    parser.add_argument('--dataset_path', type=str, default='../../dataset',
                       help='Path to dataset directory')
    
    args = parser.parse_args()
    
    # Train model
    model, result_dir = train(args)
    
    print(f"\nTraining complete! Results saved to: {result_dir}")


if __name__ == '__main__':
    main()



