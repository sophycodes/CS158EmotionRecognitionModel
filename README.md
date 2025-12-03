# CS158 Emotion Recognition Model

![Project Banner](insideOut.jpg)

**CS158 Final Project - Fall 2025**

A deep learning system for recognizing facial emotions using Convolutional Neural Networks (CNNs).

## Members
- Aiko Kato
- Bengisu Bulur  
- Sophy Figaroa

## Project Overview

This project implements a facial emotion recognition system that classifies facial expressions into 7 emotion categories:
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

The system uses Convolutional Neural Networks (CNNs) and 2 Kaggle Datasets.

**Train Dataset**
[Kaggle FER-2013](https://www.kaggle.com/datasets/msambare/fer2013), which contains approximately 49,779 grayscale images of 48x48 pixel faces.

**Test Dataset**
[Kaggle Face expression recognition dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset), which contains approximately 7,066 grayscale images of 48x48 pixel faces.

### Model Architectures

We implemented and compared three CNN architectures:

1. **SimpleCNN** (2-layer baseline)
   - 2 convolutional layers
   - Training accuracy: 97%
   - Validation accuracy: 60%
   - Shows overfitting issues

2. **MediumCNN** (3-layer with regularization)
   - 3 convolutional layers
   - Includes BatchNorm and Dropout
   - Training accuracy: 48%
   - Validation accuracy: 59%

3. **LargeCNN** (4-layer deep model)
   - 4 convolutional layers
   - gradual filter growth (64→128→256→512)
   - Training accuracy: 81%
   - Validation accuracy: 72%

### Best Performance
Our best model large_cnn_20epoch_0.01lr achieves:
   - Training accuracy: 81%
   - Validation accuracy: 72%
   - Test accuracy: 61%

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/sophycodes/CS158EmotionRecognitionModel.git
cd CS158EmotionRecognitionModel
```

### Step 2: Create Virtual Environment

Create and activate the virtual environment named `ERvenv`:

**On macOS/Linux:**
```bash
python3 -m venv ERvenv
source ERvenv/bin/activate
```

**On Windows:**
```bash
python -m venv ERvenv
ERvenv\Scripts\activate
```

You should see `(ERvenv)` appear in your terminal prompt, indicating the virtual environment is active.

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- PyTorch (torch, torchvision)
- OpenCV (opencv-python)
- NumPy
- Matplotlib
- scikit-learn
- Pillow

### Step 4: Verify Installation

```bash
python -c "import torch; import cv2; import numpy; print('Installation successful!')"
```

---
## Using Trained Models

### Downloading Pre-trained Models

Our trained models are available in the repository under the `models_results/` directory:

```
models_results/
├── large_cnn_20epoch_0.001lr         # best model

```

### Loading a Trained Model

```python
import torch
from models import LargeCNN  # or SimpleCNN, MediumCNN

# Load the model architecture
model = LargeCNN()

# Load trained weights
model.load_state_dict(torch.load('models/large_cnn.pth'))
model.eval()  # Set to evaluation mode

```

---

## Common Usage Examples

### Testing with Different Dataset Paths

**Standard test set:**
```bash
python test_model.py \
  --model_path models/medium_cnn.pth \
  --model_type medium_cnn \
  --data_dir data/test
```

**Custom dataset location:**
```bash
python test_model.py \
  --model_path models/large_cnn.pth \
  --model_type large_cnn \
  --data_dir /path/to/your/custom/test_data
```

### Expected Dataset Structure

Your `--data_dir` should contain emotion subfolders:
```
data/test/
├── angry/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── disgust/
├── fear/
├── happy/
├── neutral/
├── sad/
└── surprise/
```

---

## Running Predictions

### Testing Model on Dataset

Use test_model.py to evaluate a trained model on your test dataset:

```bash
python test_model.py \
    --model_path results/large_cnn_TIMESTAMP/model.pth \
    --model_type large_cnn \
    --data_path data/test
```

**Parameters:**
- `--model_path`: Path to your trained model (.pth file)
- `--model_type`: Architecture type (`simple_cnn`, `medium_cnn`, or `large_cnn`)
- `--data_path`: Path to test dataset directory (default: `data/test`)
- `--batch_size`: Batch size for testing (default: 64)
- `--save_results`: Save confusion matrix and classification report
- `--results_dir`: Directory to save results (default: `test_results`)

**Examples:**

Test on validation data:
```bash
python test_model.py \
    --model_path models/medium_cnn.pth \
    --model_type medium_cnn \
    --data_path data/validation \
    --save_results
```

### Live Webcam Demo

Run real-time emotion detection using your webcam:

```bash
python live_demo.py --model models/medium_cnn.pth
```

**Controls:**
- Press `q` to quit
- Press `s` to save a screenshot

**Note:** On macOS, you may need to grant camera permissions in System Preferences.

