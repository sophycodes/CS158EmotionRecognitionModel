"""
live_demo.py
Real-time emotion recognition using webcam

from video: https://www.youtube.com/watch?v=Bb4Wvl57LIk&t=1538s

Usage:
    python live_demo.py --model_path results/large_cnn_20241201_123456/model.pth
"""

import torch
import cv2
from torchvision import transforms
import argparse

# Import model from models.py
from models import LargeCNN


def load_model(model_path, device):
    """Load trained model from checkpoint"""
    model = LargeCNN(num_classes=7).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode
    print(f"Model loaded from {model_path}")
    return model


def preprocess_face(face_img):
    """
    Preprocess face image to match training preprocessing
    Grayscale -> Resize(48x48) -> ToTensor -> Normalize(0.5, 0.5)
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    return transform(face_img).unsqueeze(0)  # Add batch dimension


def predict_emotion(model, face_tensor, device, class_names):
    """Make emotion prediction"""
    with torch.no_grad():
        face_tensor = face_tensor.to(device)
        output = model(face_tensor)
        
        # Get probabilities and prediction
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        emotion = class_names[predicted.item()]
        conf = confidence.item()
        
    return emotion, conf


def run_live_demo(model_path, device, show_confidence=True):
    """Run live emotion recognition demo"""
    
    # Emotion class names (must match your dataset folder structure)
    class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    # Load model
    model = load_model(model_path, device)
    
    # Load face detection cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    #CHATGPT helped with this because we couldn't figure out how to get the webcam to work
    # Try to find working camera
    cap = None
    for camera_index in range(5):  # Try first 5 camera indices
        print(f"Trying camera index {camera_index}...")
        test_cap = cv2.VideoCapture(camera_index)
        if test_cap.isOpened():
            # Wait a moment for camera to initialize
            import time
            time.sleep(0.5)
            # Test if we can actually read a frame
            ret, _ = test_cap.read()
            if ret:
                print(f"Using camera index: {camera_index}")
                cap = test_cap
                break
            else:
                test_cap.release()
        else:
            test_cap.release()

    if cap is None:
        print("Error: Could not open any webcam")
        print("\nTroubleshooting:")
        print("1. System Settings → Privacy & Security → Camera")
        print("2. Add Terminal.app and toggle it ON")
        print("3. Close Terminal completely (Cmd+Q) and reopen")
        print("4. Close other apps using camera (Zoom, Teams, etc.)")
        return
    
    print("\n" + "="*60)
    print("LIVE EMOTION RECOGNITION DEMO")
    print("="*60)
    print("Press 'q' to quit")
    print("Press 's' to take a screenshot")
    print("="*60 + "\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame_count += 1
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48)
        )
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            # Preprocess face
            face_tensor = preprocess_face(face_roi)
            
            # Predict emotion
            emotion, confidence = predict_emotion(model, face_tensor, device, class_names)
            
            # Draw rectangle around face
            color = (0, 255, 0)  # Green
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Prepare label
            if show_confidence:
                label = f"{emotion}: {confidence:.2%}"
            else:
                label = emotion
            
            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(
                frame,
                (x, y - 35),
                (x + label_size[0], y),
                color,
                cv2.FILLED
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),  # Black text
                2
            )
        
        # Display info
        info_text = f"Faces detected: {len(faces)} | Frame: {frame_count}"
        cv2.putText(
            frame,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Show frame
        cv2.imshow('Emotion Recognition - Press q to quit, s for screenshot', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('s'):
            screenshot_name = f"screenshot_{frame_count}.jpg"
            cv2.imwrite(screenshot_name, frame)
            print(f"Screenshot saved: {screenshot_name}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Demo ended.")


def main():
    parser = argparse.ArgumentParser(description='Live emotion recognition demo')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model .pth file')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to run on (default: auto)')
    parser.add_argument('--no_confidence', action='store_true',
                       help='Hide confidence scores (show only emotion labels)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Run demo
    run_live_demo(args.model_path, device, show_confidence=not args.no_confidence)


if __name__ == '__main__':
    main()