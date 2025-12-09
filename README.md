# Object Detection Project

A collection of object detection and image classification projects using deep learning models, primarily built with YOLO (You Only Look Once) and other computer vision frameworks.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Projects](#projects)
- [Getting Started](#getting-started)
- [Object Detection Concepts](#object-detection-concepts)
- [Model Types](#model-types)
- [Requirements](#requirements)
- [Usage](#usage)
- [Contributing](#contributing)

## Overview

This repository contains multiple object detection and image classification projects. Each project focuses on detecting or classifying specific objects or categories in images using state-of-the-art deep learning models.

### What is Object Detection?

Object detection is a computer vision task that involves:
- **Localization**: Finding the location of objects in an image (bounding boxes)
- **Classification**: Identifying what type of objects are present
- **Confidence Scoring**: Providing probability scores for each detection

Unlike image classification (which identifies what's in an entire image), object detection can identify and locate multiple objects within a single image.

## Project Structure

```
Object_Detection/
│
├── README.md                          # This file - general project documentation
│
├── Mushroom_Type_Prediction/          # Mushroom classification project
│   ├── README.md                      # Project-specific documentation
│   ├── main.py                        # Main prediction script
│   ├── mushroom-prediction.ipynb      # Training notebook
│   ├── Models/                        # Trained model files
│   │   └── mushroom.pt
│   └── images/                        # Input images folder
│
└── [Future Projects]/                 # Additional projects will be added here
```

## Projects

### 1. Mushroom Type Prediction

**Type**: Image Classification  
**Model**: YOLO11 Classification Model  
**Purpose**: Classify mushroom images into different species

- **Location**: `Mushroom_Type_Prediction/`
- **Model File**: `Models/mushroom.pt`
- **Usage**: See [Mushroom_Type_Prediction/README.md](Mushroom_Type_Prediction/README.md)

**Features**:
- Processes multiple images from a folder
- Outputs predicted mushroom species with confidence scores
- Supports multiple image formats (JPG, PNG, BMP, TIFF, WebP)

---

*More projects will be added here as they are developed*

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- CUDA-capable GPU (optional, but recommended for faster inference)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Object_Detection
```

2. Install required dependencies:
```bash
pip install ultralytics
```

For specific projects, additional dependencies may be required. Check individual project README files for details.

## Object Detection Concepts

### Key Terms

- **Bounding Box**: A rectangular box that outlines the location of an object in an image
- **Confidence Score**: A probability value (0-1) indicating how certain the model is about a detection
- **Class**: The category or type of object being detected (e.g., "person", "car", "mushroom")
- **IoU (Intersection over Union)**: A metric used to evaluate detection accuracy
- **NMS (Non-Maximum Suppression)**: A technique to remove duplicate detections

### Detection vs Classification

| Feature | Object Detection | Image Classification |
|---------|-----------------|---------------------|
| Output | Bounding boxes + classes | Single class label |
| Multiple Objects | Yes | No |
| Location Info | Yes | No |
| Use Case | Finding objects in scenes | Categorizing entire images |

## Model Types

### YOLO (You Only Look Once)

YOLO is a popular real-time object detection system that processes entire images in a single pass.

**Variants**:
- **YOLOv8**: Fast and accurate detection
- **YOLOv11**: Latest version with improved performance
- **YOLO-nano**: Lightweight version for mobile/edge devices

**Modes**:
- **Detection**: Detects objects with bounding boxes
- **Classification**: Classifies entire images
- **Segmentation**: Pixel-level object segmentation
- **Pose Estimation**: Detects human poses

### Other Models

- **R-CNN Family**: Region-based CNN models (slower but more accurate)
- **SSD**: Single Shot Detector (balance between speed and accuracy)
- **RetinaNet**: Feature pyramid network for detection

## Requirements

### Core Dependencies

```txt
ultralytics>=8.0.0
torch>=1.8.0
torchvision>=0.9.0
opencv-python>=4.6.0
pillow>=7.1.2
numpy>=1.23.0
```

### Optional Dependencies

```txt
matplotlib>=3.3.0    # For visualization
pandas>=1.1.4        # For data handling
tqdm>=4.64.0         # For progress bars
```

## Usage

### Running a Project

Each project has its own main script. Navigate to the project directory and run:

```bash
cd [Project_Name]
python main.py
```

### General Workflow

1. **Prepare Data**: Place images in the designated `images/` folder
2. **Load Model**: The script automatically loads the trained model
3. **Run Inference**: Process images and get predictions
4. **View Results**: Check console output or saved result files

### Example: Mushroom Classification

```bash
cd Mushroom_Type_Prediction
# Add images to the images/ folder
python main.py
```

## Adding New Projects

To add a new object detection project:

1. Create a new folder: `[Project_Name]/`
2. Add project structure:
   ```
   [Project_Name]/
   ├── README.md          # Project documentation
   ├── main.py            # Main script
   ├── Models/            # Model files
   └── images/            # Input images
   ```
3. Update this README.md to include the new project
4. Follow the existing code structure and documentation style

### Project Template

```python
# main.py template
from ultralytics import YOLO
from pathlib import Path

def main():
    # Load model
    model = YOLO('Models/model.pt')
    
    # Process images
    images_folder = Path('images')
    for image_path in images_folder.glob('*.jpg'):
        results = model(str(image_path))
        # Process results
        print(f"Results for {image_path.name}")

if __name__ == "__main__":
    main()
```

## Model Training

For training new models, refer to individual project notebooks or training scripts. General training steps:

1. **Data Preparation**: Organize images into train/val/test splits
2. **Annotation**: Label objects (for detection) or organize by class (for classification)
3. **Training**: Run training script with appropriate hyperparameters
4. **Validation**: Evaluate model performance on validation set
5. **Export**: Save trained model for inference

## Performance Metrics

Common metrics used in object detection:

- **mAP (mean Average Precision)**: Overall detection accuracy
- **Precision**: Ratio of correct detections to total detections
- **Recall**: Ratio of detected objects to total objects
- **F1-Score**: Harmonic mean of precision and recall
- **Inference Speed**: FPS (frames per second) for real-time applications

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure model file exists in `Models/` folder
2. **No images found**: Check that images are in the correct folder with supported formats
3. **CUDA errors**: Install CUDA-compatible PyTorch or use CPU mode
4. **Memory errors**: Reduce batch size or image resolution

### Getting Help

- Check individual project README files
- Review model documentation: [Ultralytics Docs](https://docs.ultralytics.com)
- Check error messages for specific guidance

## Contributing

When contributing:

1. Follow the existing project structure
2. Add comprehensive documentation
3. Include example usage
4. Test with sample images
5. Update this README with new projects

## License

[Specify your license here]

## Acknowledgments

- [Ultralytics](https://ultralytics.com) for YOLO framework
- Open source computer vision community

## Resources

- [YOLO Documentation](https://docs.ultralytics.com)
- [Computer Vision Tutorials](https://pytorch.org/tutorials/beginner/introyt/introyt1_tutorial.html)
- [Object Detection Papers](https://github.com/amusi/awesome-object-detection)

---

**Last Updated**: [Current Date]  
**Maintained by**: [Your Name/Team]

