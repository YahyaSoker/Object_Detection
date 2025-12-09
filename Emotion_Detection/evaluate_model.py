import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from tqdm import tqdm
import platform
import pandas as pd

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuration
DATA_DIR = 'Data'
MODEL_DIR = 'models'
RESULTS_DIR = 'results'
MODEL_PATH = os.path.join(MODEL_DIR, 'emotion_model_best.pth')
IMAGE_SIZE = 256
BATCH_SIZE = 32

# Windows compatibility: num_workers=0 on Windows to avoid multiprocessing issues
NUM_WORKERS = 0 if platform.system() == 'Windows' else 4

# Default emotion classes (will be loaded from model checkpoint)
DEFAULT_CLASSES = ['Angry', 'Fear', 'Happy', 'Sad', 'Suprise']

def load_model(model_path=MODEL_PATH, device=device):
    """Load the trained model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    
    print(f"Loading model from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get class names from checkpoint or use default
    class_names = checkpoint.get('class_names', DEFAULT_CLASSES)
    
    # Get model info
    epoch = checkpoint.get('epoch', 'Unknown')
    val_acc = checkpoint.get('val_acc', 'Unknown')
    val_loss = checkpoint.get('val_loss', 'Unknown')
    
    print(f"Model trained for {epoch} epochs")
    print(f"Validation accuracy: {val_acc:.2f}%" if isinstance(val_acc, (int, float)) else f"Validation accuracy: {val_acc}")
    print(f"Validation loss: {val_loss:.4f}" if isinstance(val_loss, (int, float)) else f"Validation loss: {val_loss}")
    
    # Create model
    model = models.resnet50(weights=None)  # Don't load ImageNet weights
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(class_names))
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Classes: {class_names}")
    
    return model, class_names

def get_test_loader(data_dir=DATA_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE):
    """Create data loader for test set."""
    
    # Test transform (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create test dataset
    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'test'),
        transform=test_transform
    )
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return test_loader, test_dataset.classes

def evaluate_model(model, test_loader, device, class_names):
    """Evaluate the model on test set."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    running_loss = 0.0
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss()
    
    print("\nEvaluating on test set...")
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluation")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Get predictions
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            # Statistics
            running_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100 * correct / total
    
    return all_labels, all_preds, all_probs, test_loss, test_acc

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot confusion matrix with normalized values."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_xlabel('Predicted', fontsize=12)
    axes[0].set_ylabel('Actual', fontsize=12)
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    
    # Normalized confusion matrix
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar_kws={'label': 'Proportion'})
    axes[1].set_xlabel('Predicted', fontsize=12)
    axes[1].set_ylabel('Actual', fontsize=12)
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")

def plot_class_performance(y_true, y_pred, class_names, save_path):
    """Plot per-class performance metrics."""
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )
    
    # Create DataFrame for easier plotting
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(class_names))
    width = 0.25
    
    ax.bar(x - width, precision, width, label='Precision', color='#3498db')
    ax.bar(x, recall, width, label='Recall', color='#2ecc71')
    ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')
    
    ax.set_xlabel('Emotion Classes', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    # Add value labels on bars
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        ax.text(i - width, p + 0.02, f'{p:.2f}', ha='center', va='bottom', fontsize=9)
        ax.text(i, r + 0.02, f'{r:.2f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + width, f + 0.02, f'{f:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Class performance plot saved to: {save_path}")
    
    return metrics_df

def save_evaluation_report(y_true, y_pred, class_names, test_loss, test_acc, metrics_df, save_path):
    """Save comprehensive evaluation report."""
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=False)
    
    with open(save_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("MODEL EVALUATION REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write("OVERALL METRICS\n")
        f.write("-"*60 + "\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.2f}%\n")
        f.write(f"Total Test Samples: {len(y_true)}\n")
        f.write("\n")
        
        f.write("CLASSIFICATION REPORT\n")
        f.write("-"*60 + "\n")
        f.write(report)
        f.write("\n")
        
        f.write("PER-CLASS METRICS\n")
        f.write("-"*60 + "\n")
        f.write(metrics_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("="*60 + "\n")
    
    print(f"Evaluation report saved to: {save_path}")

def print_summary(y_true, y_pred, class_names, test_loss, test_acc):
    """Print evaluation summary."""
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0, average=None
    )
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Total Test Samples: {len(y_true)}")
    
    print("\n" + "-"*60)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-"*60)
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")
    
    # Overall metrics
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()
    weighted_precision = np.average(precision, weights=support)
    weighted_recall = np.average(recall, weights=support)
    weighted_f1 = np.average(f1, weights=support)
    
    print("-"*60)
    print(f"{'Macro Avg':<15} {macro_precision:<12.4f} {macro_recall:<12.4f} {macro_f1:<12.4f} {support.sum():<10}")
    print(f"{'Weighted Avg':<15} {weighted_precision:<12.4f} {weighted_recall:<12.4f} {weighted_f1:<12.4f} {support.sum():<10}")
    print("="*60)

def main():
    """Main evaluation function."""
    print("="*60)
    print("MODEL EVALUATION")
    print("="*60)
    print(f"Device: {device}")
    print(f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print("="*60)
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load model
    try:
        model, class_names = load_model(MODEL_PATH, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Get test loader
    try:
        test_loader, dataset_classes = get_test_loader()
        print(f"\nTest dataset loaded")
        print(f"Test samples: {len(test_loader.dataset)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Verify class names match
        if dataset_classes != class_names:
            print(f"\nWarning: Dataset classes {dataset_classes} don't match model classes {class_names}")
            print("Using model classes for evaluation")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return
    
    # Evaluate model
    try:
        y_true, y_pred, y_probs, test_loss, test_acc = evaluate_model(
            model, test_loader, device, class_names
        )
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return
    
    # Print summary
    print_summary(y_true, y_pred, class_names, test_loss, test_acc)
    
    # Generate visualizations and reports
    print("\nGenerating evaluation metrics...")
    
    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names,
                         os.path.join(RESULTS_DIR, 'test_confusion_matrix.png'))
    
    # Per-class performance
    metrics_df = plot_class_performance(y_true, y_pred, class_names,
                                       os.path.join(RESULTS_DIR, 'test_class_performance.png'))
    
    # Save evaluation report
    save_evaluation_report(y_true, y_pred, class_names, test_loss, test_acc, metrics_df,
                          os.path.join(RESULTS_DIR, 'test_evaluation_report.txt'))
    
    print("\n" + "="*60)
    print("Evaluation completed!")
    print("="*60)
    print(f"\nAll results saved to '{RESULTS_DIR}/' directory:")
    print(f"  - test_confusion_matrix.png")
    print(f"  - test_class_performance.png")
    print(f"  - test_evaluation_report.txt")

if __name__ == '__main__':
    main()

