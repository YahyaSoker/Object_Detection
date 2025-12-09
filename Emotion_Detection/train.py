import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
import platform

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configuration
DATA_DIR = 'Data'
MODEL_DIR = 'models'
RESULTS_DIR = 'results'
IMAGE_SIZE = 256
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
NUM_CLASSES = 5
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.001

# Windows compatibility: num_workers=0 on Windows to avoid multiprocessing issues
NUM_WORKERS = 0 if platform.system() == 'Windows' else 4

# Create directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Emotion classes (must match folder names)
EMOTION_CLASSES = ['Angry', 'Fear', 'Happy', 'Sad', 'Suprise']

def get_data_loaders():
    """Create data loaders for train and validation sets."""
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # No augmentation for validation
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_DIR, 'train'),
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_DIR, 'valid'),
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, train_dataset.classes

def create_model(num_classes=NUM_CLASSES):
    """Create ResNet50 model with modified final layer."""
    
    # Load pre-trained ResNet50
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    
    # Freeze early layers (optional - can be unfrozen for fine-tuning)
    # for param in model.parameters():
    #     param.requires_grad = False
    
    # Modify final layer for our number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device, return_predictions=False):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if return_predictions:
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    if return_predictions:
        return epoch_loss, epoch_acc, all_preds, all_labels
    return epoch_loss, epoch_acc

def plot_training_history(train_losses, train_accs, val_losses, val_accs, save_path):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to: {save_path}")

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")

def save_classification_report(y_true, y_pred, class_names, save_path):
    """Save classification report."""
    report = classification_report(y_true, y_pred, target_names=class_names, 
                                  output_dict=False)
    
    with open(save_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("="*60 + "\n\n")
        f.write(report)
    
    print(f"Classification report saved to: {save_path}")

def main():
    """Main training function."""
    print("="*60)
    print("EMOTION DETECTION MODEL TRAINING")
    print("="*60)
    print(f"Device: {device}")
    print(f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Number of Epochs: {NUM_EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print("="*60)
    
    # Get data loaders
    print("\nLoading data...")
    train_loader, val_loader, class_names = get_data_loaders()
    print(f"Classes: {class_names}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(num_classes=len(class_names))
    model = model.to(device)
    print(f"Model created: ResNet50")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Training history
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # Early stopping
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None
    final_epoch = NUM_EPOCHS
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    # Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save history
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Print epoch summary
        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping and model checkpointing
        if val_loss < best_val_loss - EARLY_STOPPING_MIN_DELTA:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'class_names': class_names,
            }, os.path.join(MODEL_DIR, 'emotion_model_best.pth'))
            print(f"✓ New best model saved! (Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                final_epoch = epoch
                print(f"\nEarly stopping triggered after {epoch} epochs")
                print(f"Best validation loss: {best_val_loss:.4f}")
                print(f"Best validation accuracy: {best_val_acc:.2f}%")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nBest model loaded (Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.2f}%)")
    
    # Save final model
    torch.save({
        'epoch': final_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
        'class_names': class_names,
    }, os.path.join(MODEL_DIR, 'emotion_model_last.pth'))
    print(f"Final model saved to {os.path.join(MODEL_DIR, 'emotion_model_last.pth')}")
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print("="*60)
    
    # Generate final validation predictions for evaluation
    print("\nGenerating evaluation metrics...")
    _, _, final_preds, final_labels = validate(model, val_loader, criterion, device, return_predictions=True)
    
    # Plot training history
    plot_training_history(train_losses, train_accs, val_losses, val_accs,
                          os.path.join(RESULTS_DIR, 'training_history.png'))
    
    # Plot confusion matrix
    plot_confusion_matrix(final_labels, final_preds, class_names,
                          os.path.join(RESULTS_DIR, 'confusion_matrix.png'))
    
    # Save classification report
    save_classification_report(final_labels, final_preds, class_names,
                               os.path.join(RESULTS_DIR, 'classification_report.txt'))
    
    print("\nAll results saved to 'results/' directory")
    print("Model saved to 'models/' directory")

if __name__ == '__main__':
    main()

