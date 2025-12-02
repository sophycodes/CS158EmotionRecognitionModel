"""
models.py
All model architectures in one place
"""
import torch
import torch.nn as nn

class BasicNN(nn.Module):
    """Baseline 2-layer feedforward network"""
    def __init__(self, num_classes=7):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 48, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)


class SimpleCNN(nn.Module):
    """
    Simple 2-layer CNN 
    NO regularization at all
    
    """
    def __init__(self, num_classes=7):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 12 * 12, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)


class MediumCNN(nn.Module):
    """3-layer CNN with BatchNorm and Dropout"""
    
    def __init__(self, num_classes=7):  
        super().__init__()
        
        # create a container that will execute layers in order
        self.layers = nn.Sequential( 
                                    
            # Convolutional layer:
            # - 1 = input channels (grayscale images have 1 channel; RGB would be 3)
            # - 32 = output channels (number of filters/feature maps to learn) 
            # - kernel_size=3 = each filter is a 3×3 grid of weights
            # - padding=1 = adds 1 pixel of zeros around the image border
            
            # Block 1: 32 filters
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # normalizes values across batch to have mean≈0, std≈1
            nn.ReLU(),  # activation function - adds non-linearity
            nn.MaxPool2d(2),  # downsamples by taking max value in each 2x2 block
            nn.Dropout(0.25),  # randomly turns off 25% of neurons during training
            
            # Block 2: 64 filters
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            # Block 3: 128 filters
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            # Flatten and fully connected layers
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """Forward pass through the network"""
        return self.layers(x)
    
    
class LargeCNN(nn.Module):
    """
    4-layer CNN with BatchNorm and Dropout
    Gradual filter growth: 64→128→256→512
    Higher capacity than MediumCNN for better performance
    """
    def __init__(self, num_classes=7):
        super().__init__()
        
        self.layers = nn.Sequential(
            # Block 1: 64 filters
            # Starts with more filters than MediumCNN for increased capacity
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 48x48 -> 24x24
            nn.Dropout(0.2),  # Lower dropout - not overfitting
            
            # Block 2: 128 filters
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 24x24 -> 12x12
            nn.Dropout(0.25),
            
            # Block 3: 256 filters
            # Gradual growth prevents training instability
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 12x12 -> 6x6
            nn.Dropout(0.25),
            
            # Block 4: 512 filters (NEW - adds depth)
            # This 4th layer is what gives the model extra capacity
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 6x6 -> 3x3
            nn.Dropout(0.3),
            
            # Flatten and fully connected layers
            nn.Flatten(),
            # 512 filters * 3x3 spatial dimensions = 4608 features
            nn.Linear(512 * 3 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),  # Higher dropout in FC layers where overfitting typically occurs
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """Forward pass through the network"""
        return self.layers(x)
    