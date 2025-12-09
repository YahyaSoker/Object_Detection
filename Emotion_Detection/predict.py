import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from PIL import Image
import argparse
import os
from pathlib import Path

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuration
MODEL_PATH = 'models/emotion_model_best.pth'
IMAGE_SIZE = 256
NUM_CLASSES = 5

# Default emotion classes (will be loaded from model checkpoint)
DEFAULT_CLASSES = ['Angry', 'Fear', 'Happy', 'Sad', 'Suprise']

def load_model(model_path=MODEL_PATH, device=device):
    """Load the trained model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get class names from checkpoint or use default
    class_names = checkpoint.get('class_names', DEFAULT_CLASSES)
    
    # Create model
    model = models.resnet50(weights=None)  # Don't load ImageNet weights
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(class_names))
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, class_names

def preprocess_image(image_path, image_size=IMAGE_SIZE):
    """Preprocess image for inference."""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Define transform (same as validation)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transform
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor, image

def predict_emotion(model, image_tensor, class_names, device=device):
    """Predict emotion from image tensor."""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        # Get all class probabilities
        all_probs = probabilities[0].cpu().numpy()
    
    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item()
    
    return predicted_class, confidence_score, all_probs

def display_results(image_path, predicted_class, confidence_score, all_probs, class_names):
    """Display prediction results."""
    print("\n" + "="*60)
    print("EMOTION PREDICTION RESULTS")
    print("="*60)
    print(f"Image: {image_path}")
    print(f"\nPredicted Emotion: {predicted_class}")
    print(f"Confidence: {confidence_score*100:.2f}%")
    print("\n" + "-"*60)
    print("Confidence Scores for All Classes:")
    print("-"*60)
    
    # Sort by probability
    sorted_indices = sorted(range(len(all_probs)), key=lambda i: all_probs[i], reverse=True)
    
    for idx in sorted_indices:
        bar_length = int(all_probs[idx] * 40)
        bar = '█' * bar_length + '░' * (40 - bar_length)
        print(f"{class_names[idx]:<15} {all_probs[idx]*100:>6.2f}% {bar}")
    
    print("="*60)

def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='Predict emotion from a single image')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('--model', type=str, default=MODEL_PATH,
                       help=f'Path to the trained model (default: {MODEL_PATH})')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        return
    
    print(f"Using device: {device}")
    print(f"Loading model from: {args.model}")
    
    # Load model
    try:
        model, class_names = load_model(args.model, device)
        print(f"Model loaded successfully!")
        print(f"Classes: {class_names}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Preprocess image
    try:
        image_tensor, original_image = preprocess_image(args.image_path)
        print(f"Image loaded and preprocessed: {original_image.size}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Predict
    predicted_class, confidence_score, all_probs = predict_emotion(
        model, image_tensor, class_names, device
    )
    
    # Display results
    display_results(args.image_path, predicted_class, confidence_score, 
                   all_probs, class_names)

if __name__ == '__main__':
    main()

