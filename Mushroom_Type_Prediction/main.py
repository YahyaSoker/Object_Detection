import os
from pathlib import Path
from ultralytics import YOLO

def main():
    # Load the trained model
    model_path = Path(__file__).parent / "Models" / "mushroom.pt"
    print(f"Loading model from: {model_path}")
    model = YOLO(str(model_path))
    print("Model loaded successfully!\n")
    
    # Get images folder path
    images_folder = Path(__file__).parent / "images"
    
    # Supported image formats
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    # Find all image files
    image_files = []
    if images_folder.exists():
        for ext in image_extensions:
            image_files.extend(images_folder.glob(f"*{ext}"))
            image_files.extend(images_folder.glob(f"*{ext.upper()}"))
    else:
        print(f"Images folder not found: {images_folder}")
        return
    
    if not image_files:
        print(f"No images found in: {images_folder}")
        print("Please add image files to the images folder.")
        return
    
    print(f"Found {len(image_files)} image(s) to process\n")
    print("=" * 60)
    
    # Process each image
    for image_path in sorted(image_files):
        try:
            # Run prediction
            results = model(str(image_path), verbose=False)
            
            # Get the top prediction
            if results[0].probs is not None:
                probs = results[0].probs.data.cpu().numpy()
                top1_idx = int(results[0].probs.top1)
                confidence = float(results[0].probs.top1conf)
                class_name = model.names[top1_idx]
                
                # Print result
                print(f"Image: {image_path.name}")
                print(f"Predicted Class: {class_name}")
                print(f"Confidence: {confidence:.2%}")
                print("-" * 60)
            else:
                print(f"Image: {image_path.name}")
                print("No prediction available")
                print("-" * 60)
                
        except Exception as e:
            print(f"Error processing {image_path.name}: {str(e)}")
            print("-" * 60)

if __name__ == "__main__":
    main()

