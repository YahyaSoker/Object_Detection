# Mushroom Type Prediction Script

## Overview

This script uses a pre-trained YOLO (You Only Look Once) classification model to predict mushroom types from images. It processes all images in the `images` folder and outputs the predicted mushroom class along with confidence scores.

## Requirements

- Python 3.x
- `ultralytics` library (for YOLO model)
- Trained model file: `Models/mushroom.pt`

## Installation

Install the required dependency:

```bash
pip install ultralytics
```

## Code Explanation

### Imports

```python
import os
from pathlib import Path
from ultralytics import YOLO
```

- `os`: Operating system interface (used implicitly)
- `pathlib.Path`: Object-oriented filesystem paths for cross-platform compatibility
- `ultralytics.YOLO`: The YOLO model class for loading and running predictions

### Main Function

The `main()` function performs the following steps:

#### 1. Load the Model

```python
model_path = Path(__file__).parent / "Models" / "mushroom.pt"
model = YOLO(str(model_path))
```

- Constructs the path to the model file relative to the script location
- Loads the pre-trained YOLO classification model from `Models/mushroom.pt`
- The model contains learned weights for classifying mushroom images into different species

#### 2. Locate Images Folder

```python
images_folder = Path(__file__).parent / "images"
```

- Gets the path to the `images` folder in the same directory as the script
- Uses `Path(__file__).parent` to ensure the script works regardless of where it's run from

#### 3. Find Image Files

```python
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
image_files = []
for ext in image_extensions:
    image_files.extend(images_folder.glob(f"*{ext}"))
    image_files.extend(images_folder.glob(f"*{ext.upper()}"))
```

- Defines supported image formats
- Searches for all image files with these extensions (case-insensitive)
- Collects all matching files into a list

#### 4. Process Each Image

```python
for image_path in sorted(image_files):
    results = model(str(image_path), verbose=False)
```

- Iterates through all found images (sorted alphabetically)
- Runs the model prediction on each image
- `verbose=False` suppresses model output messages

#### 5. Extract Predictions

```python
top1_idx = int(results[0].probs.top1)
confidence = float(results[0].probs.top1conf)
class_name = model.names[top1_idx]
```

- `top1_idx`: Index of the most likely class
- `confidence`: Confidence score (0-1) for the top prediction
- `class_name`: Human-readable name of the predicted mushroom species from the model's class dictionary

#### 6. Display Results

```python
print(f"Image: {image_path.name}")
print(f"Predicted Class: {class_name}")
print(f"Confidence: {confidence:.2%}")
```

- Prints the image filename, predicted mushroom class, and confidence percentage
- Results are separated by a line for readability

### Error Handling

The script includes error handling for:
- Missing `images` folder
- No images found in the folder
- Errors during image processing (corrupted files, unsupported formats, etc.)

## Usage

1. Place mushroom images in the `images` folder
2. Ensure the model file exists at `Models/mushroom.pt`
3. Run the script:

```bash
python main.py
```

## Output Format

The script outputs results in the following format:

```
Loading model from: [path/to/model]
Model loaded successfully!

Found X image(s) to process

============================================================
Image: mushroom1.jpg
Predicted Class: Amanita muscaria
Confidence: 95.23%
------------------------------------------------------------
Image: mushroom2.png
Predicted Class: Boletus edulis
Confidence: 87.45%
------------------------------------------------------------
```

## Supported Image Formats

- JPEG (`.jpg`, `.jpeg`)
- PNG (`.png`)
- BMP (`.bmp`)
- TIFF (`.tiff`)
- WebP (`.webp`)

## Notes

- The script processes images sequentially
- Each image is analyzed independently
- Confidence scores indicate how certain the model is about its prediction
- Higher confidence scores (closer to 100%) indicate more reliable predictions

