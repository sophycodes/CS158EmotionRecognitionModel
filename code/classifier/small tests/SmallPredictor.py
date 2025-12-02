"""
SmallPredictor.py
Use the small CNN (trained on 10 images per emotion) to predict emotion from a single input image
Run python SmallPredictor.py path/to/image.jpg

# We can do normal gradient descent 
Loss Function: cross-entropy loss
Algorithm: ADAM (gradient descent w/two moving averages for each weight Average of recent gradients (momentum) + Average of recent squared gradients)
"""



import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import os
import sys

# configuration

img_size = 48  # model expects 48x48 grayscale images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

# same class ordering used during training
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


# define model (same as SmallDataTrainer.py)

class small_cnn(nn.Module):
    def __init__(self, num_classes):
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


# load model

model_path = "small_emotion_cnn.pth"  # saved model file

if not os.path.exists(model_path):  # check if file exists
    print(f"ERROR: model file not found: {model_path}")
    sys.exit(1)

# create the model and load learned weights
model = small_cnn(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # set model to evaluation mode

print("loaded model:", model_path)


# image transform (same as training)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # convert to grayscale
    transforms.Resize((img_size, img_size)),  # resize to 48×48
    transforms.ToTensor(),  # convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # normalize to [-1,1]
])


# predict function (load image, preprocess it, run model, and overlay prediction)

def predict_emotion(image_path):  # ensure file exists
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        sys.exit(1)

    # load original image with OpenCV (for display + drawing text)
    original = cv2.imread(image_path)
    if original is None:
        print("ERROR: Could not read image with cv2.")
        sys.exit(1)

    # also load image via PIL for preprocessing
    pil_img = Image.open(image_path).convert("RGB")  # ensure RGB format

    # apply same preprocessing as training
    tensor = transform(pil_img)
    tensor = tensor.unsqueeze(0).to(device)

    # prediction step (no gradient needed)
    with torch.no_grad():
        outputs = model(tensor)
        pred_label = torch.argmax(outputs, dim=1).item()

    emotion = class_names[pred_label]
    print(f"Predicted emotion: {emotion}")


    # draw prediction text on original image
    
    # make editable copy
    output_img = original.copy()

    # height, width
    h, w = output_img.shape[:2]

    # text position near top-left
    text_pos = (10, int(h * 0.1))

    # auto-scale text size based on image resolution
    font_scale = max(0.6, min(w, h) / 300)

    # draw text label
    cv2.putText(
        output_img,
        emotion,
        text_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,  # automatically scaled font size
        (0, 255, 0),  # green text
        2,  # text thickness
        cv2.LINE_AA
    )

    # save result
    save_path = "predicted_output.jpg"
    cv2.imwrite(save_path, output_img)
    print(f"Saved output image as {save_path}")

    # show
    cv2.imshow("prediction", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# main to handle command-line arguments

if __name__ == "__main__":
    if len(sys.argv) < 2:  # require image path
        print("Usage: python SmallPredictor.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]  # read image from CLI
    predict_emotion(img_path)  # run prediction