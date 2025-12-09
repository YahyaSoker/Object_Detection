import os
import shutil
import random
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def split_data(source_dir='Data', train_ratio=0.7, test_ratio=0.15, valid_ratio=0.15):
    """
    Split emotion data into train, test, and validation sets.
    
    Args:
        source_dir: Directory containing emotion class folders
        train_ratio: Proportion of data for training
        test_ratio: Proportion of data for testing
        valid_ratio: Proportion of data for validation
    """
    # Verify ratios sum to 1
    assert abs(train_ratio + test_ratio + valid_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    source_path = Path(source_dir)
    
    # Get all emotion class directories
    emotion_classes = [d.name for d in source_path.iterdir() if d.is_dir() and d.name not in ['train', 'test', 'valid']]
    emotion_classes.sort()
    
    print(f"Found emotion classes: {emotion_classes}")
    
    # Create output directories
    for split in ['train', 'test', 'valid']:
        split_dir = source_path / split
        split_dir.mkdir(exist_ok=True)
        for emotion in emotion_classes:
            (split_dir / emotion).mkdir(exist_ok=True)
    
    # Statistics dictionary
    stats = {
        'total': {},
        'train': {},
        'test': {},
        'valid': {}
    }
    
    # Process each emotion class
    for emotion in emotion_classes:
        emotion_path = source_path / emotion
        
        # Get all image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
        image_files = [f for f in emotion_path.iterdir() 
                      if f.is_file() and f.suffix in image_extensions]
        
        total_images = len(image_files)
        stats['total'][emotion] = total_images
        
        # Shuffle images randomly
        random.shuffle(image_files)
        
        # Calculate split indices
        train_end = int(total_images * train_ratio)
        test_end = train_end + int(total_images * test_ratio)
        
        # Split files
        train_files = image_files[:train_end]
        test_files = image_files[train_end:test_end]
        valid_files = image_files[test_end:]
        
        # Move files to respective directories
        for file in train_files:
            dest = source_path / 'train' / emotion / file.name
            shutil.move(str(file), str(dest))
        
        for file in test_files:
            dest = source_path / 'test' / emotion / file.name
            shutil.move(str(file), str(dest))
        
        for file in valid_files:
            dest = source_path / 'valid' / emotion / file.name
            shutil.move(str(file), str(dest))
        
        stats['train'][emotion] = len(train_files)
        stats['test'][emotion] = len(test_files)
        stats['valid'][emotion] = len(valid_files)
        
        print(f"\n{emotion}:")
        print(f"  Total: {total_images}")
        print(f"  Train: {len(train_files)} ({len(train_files)/total_images*100:.1f}%)")
        print(f"  Test: {len(test_files)} ({len(test_files)/total_images*100:.1f}%)")
        print(f"  Valid: {len(valid_files)} ({len(valid_files)/total_images*100:.1f}%)")
    
    return stats, emotion_classes

def create_pie_chart(stats, emotion_classes, save_path='results/data_distribution.png'):
    """
    Create pie chart showing class distribution.
    
    Args:
        stats: Statistics dictionary from split_data
        emotion_classes: List of emotion class names
        save_path: Path to save the pie chart
    """
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Calculate total images per class
    class_totals = [stats['total'][emotion] for emotion in emotion_classes]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Pie chart for overall distribution
    colors = plt.cm.Set3(np.linspace(0, 1, len(emotion_classes)))
    axes[0].pie(class_totals, labels=emotion_classes, autopct='%1.1f%%', 
                startangle=90, colors=colors)
    axes[0].set_title('Overall Class Distribution', fontsize=14, fontweight='bold')
    
    # Bar chart for split distribution
    x = np.arange(len(emotion_classes))
    width = 0.25
    
    train_counts = [stats['train'][emotion] for emotion in emotion_classes]
    test_counts = [stats['test'][emotion] for emotion in emotion_classes]
    valid_counts = [stats['valid'][emotion] for emotion in emotion_classes]
    
    axes[1].bar(x - width, train_counts, width, label='Train', color='#3498db')
    axes[1].bar(x, test_counts, width, label='Test', color='#e74c3c')
    axes[1].bar(x + width, valid_counts, width, label='Validation', color='#2ecc71')
    
    axes[1].set_xlabel('Emotion Classes', fontsize=12)
    axes[1].set_ylabel('Number of Images', fontsize=12)
    axes[1].set_title('Data Split Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(emotion_classes, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    plt.close()

def print_statistics(stats, emotion_classes):
    """
    Print detailed statistics about the dataset.
    
    Args:
        stats: Statistics dictionary from split_data
        emotion_classes: List of emotion class names
    """
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    total_all = sum(stats['total'].values())
    train_all = sum(stats['train'].values())
    test_all = sum(stats['test'].values())
    valid_all = sum(stats['valid'].values())
    
    print(f"\nTotal Images: {total_all:,}")
    print(f"  Train: {train_all:,} ({train_all/total_all*100:.1f}%)")
    print(f"  Test: {test_all:,} ({test_all/total_all*100:.1f}%)")
    print(f"  Validation: {valid_all:,} ({valid_all/total_all*100:.1f}%)")
    
    print("\n" + "-"*60)
    print(f"{'Class':<15} {'Total':<10} {'Train':<10} {'Test':<10} {'Valid':<10}")
    print("-"*60)
    
    for emotion in emotion_classes:
        total = stats['total'][emotion]
        train = stats['train'][emotion]
        test = stats['test'][emotion]
        valid = stats['valid'][emotion]
        print(f"{emotion:<15} {total:<10} {train:<10} {test:<10} {valid:<10}")
    
    print("="*60)

def main():
    """Main function to prepare data."""
    print("Starting data preparation...")
    print("="*60)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Split data
    stats, emotion_classes = split_data()
    
    # Print statistics
    print_statistics(stats, emotion_classes)
    
    # Create visualizations
    create_pie_chart(stats, emotion_classes)
    
    print("\nData preparation completed successfully!")
    print(f"Data has been split into train/, test/, and valid/ directories in Data/")

if __name__ == '__main__':
    main()

