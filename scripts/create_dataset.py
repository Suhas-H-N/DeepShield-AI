"""
Dataset Creation and Preprocessing Script
Downloads and prepares deepfake datasets for training
"""

import os
import sys
from pathlib import Path
import argparse
import shutil
from tqdm import tqdm
import json
import cv2
import numpy as np
from PIL import Image

sys.path.append('../backend')
from utils.face_detector import FaceDetector
from utils.preprocessing import preprocess_image


def create_directory_structure(base_dir):
    """Create dataset directory structure"""
    base_path = Path(base_dir)
    
    directories = [
        'train/real',
        'train/fake',
        'val/real',
        'val/fake',
        'test/real',
        'test/fake',
        'samples'
    ]
    
    for directory in directories:
        (base_path / directory).mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Created directory structure at {base_dir}")


def extract_faces_from_images(input_dir, output_dir, face_detector):
    """
    Extract faces from images in a directory
    
    Args:
        input_dir: Input directory with images
        output_dir: Output directory for extracted faces
        face_detector: FaceDetector instance
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))
    
    processed = 0
    skipped = 0
    
    for img_file in tqdm(image_files, desc=f"Processing {input_path.name}"):
        try:
            # Read image
            image = cv2.imread(str(img_file))
            if image is None:
                skipped += 1
                continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            faces = face_detector.detect_faces(image_rgb)
            
            if len(faces) == 0:
                skipped += 1
                continue
            
            # Extract largest face
            face = faces[0]
            face_crop = face_detector.extract_face(image_rgb, face, target_size=(224, 224))
            
            # Save face
            output_file = output_path / f"{img_file.stem}_face.jpg"
            face_pil = Image.fromarray(face_crop)
            face_pil.save(output_file, quality=95)
            
            processed += 1
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            skipped += 1
    
    print(f"✓ Processed {processed} images, skipped {skipped}")


def split_dataset(source_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split dataset into train/val/test sets
    
    Args:
        source_dir: Source directory with real and fake subdirectories
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
    """
    source_path = Path(source_dir)
    
    for class_name in ['real', 'fake']:
        class_dir = source_path / class_name
        
        if not class_dir.exists():
            print(f"Warning: {class_dir} does not exist")
            continue
        
        # Get all images
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        
        # Shuffle
        np.random.shuffle(images)
        
        # Calculate split indices
        n = len(images)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        # Split
        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]
        
        # Copy to respective directories
        for split_name, split_images in [('train', train_images), 
                                         ('val', val_images), 
                                         ('test', test_images)]:
            dest_dir = source_path.parent / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            for img in tqdm(split_images, desc=f"{split_name}/{class_name}"):
                shutil.copy(img, dest_dir / img.name)
        
        print(f"✓ {class_name}: {len(train_images)} train, "
              f"{len(val_images)} val, {len(test_images)} test")


def download_sample_datasets():
    """Download sample datasets (placeholder)"""
    print("\n" + "="*60)
    print("DATASET DOWNLOAD INSTRUCTIONS")
    print("="*60)
    
    datasets = {
        "FaceForensics++": "https://github.com/ondyari/FaceForensics",
        "Celeb-DF": "https://github.com/yuezunli/celeb-deepfakeforensics",
        "DFDC": "https://deepfakedetectionchallenge.ai/",
        "DeeperForensics": "https://github.com/EndlessSora/DeeperForensics-1.0"
    }
    
    print("\nRecommended Datasets:")
    print("-" * 60)
    for name, url in datasets.items():
        print(f"\n{name}:")
        print(f"  URL: {url}")
    
    print("\n" + "="*60)
    print("\nDownload Instructions:")
    print("1. Visit the URLs above and follow their download instructions")
    print("2. Extract datasets to appropriate folders:")
    print("   - Real images → data/raw/real/")
    print("   - Fake images → data/raw/fake/")
    print("3. Run this script with --preprocess flag to extract faces")
    print("="*60 + "\n")


def analyze_dataset(data_dir):
    """Analyze dataset statistics"""
    data_path = Path(data_dir)
    
    stats = {
        'train': {'real': 0, 'fake': 0},
        'val': {'real': 0, 'fake': 0},
        'test': {'real': 0, 'fake': 0}
    }
    
    for split in ['train', 'val', 'test']:
        for class_name in ['real', 'fake']:
            class_dir = data_path / split / class_name
            if class_dir.exists():
                count = len(list(class_dir.glob('*.jpg'))) + len(list(class_dir.glob('*.png')))
                stats[split][class_name] = count
    
    # Print statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        real = stats[split]['real']
        fake = stats[split]['fake']
        total = real + fake
        
        print(f"\n{split.upper()}:")
        print(f"  Real: {real:,}")
        print(f"  Fake: {fake:,}")
        print(f"  Total: {total:,}")
        
        if total > 0:
            print(f"  Balance: {real/total*100:.1f}% real, {fake/total*100:.1f}% fake")
    
    total_samples = sum(stats[split]['real'] + stats[split]['fake'] 
                       for split in ['train', 'val', 'test'])
    
    print(f"\nGRAND TOTAL: {total_samples:,} samples")
    print("="*60 + "\n")
    
    # Save statistics
    stats_file = data_path / 'dataset_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✓ Statistics saved to {stats_file}")


def create_sample_dataset(output_dir, num_samples=100):
    """
    Create a small sample dataset for testing
    
    Args:
        output_dir: Output directory
        num_samples: Number of sample images per class
    """
    output_path = Path(output_dir)
    
    print("\n" + "="*60)
    print("CREATING SAMPLE DATASET")
    print("="*60 + "\n")
    
    print(f"Generating {num_samples} synthetic samples per class...")
    
    for split in ['train', 'val', 'test']:
        for class_name in ['real', 'fake']:
            split_dir = output_path / split / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine number of samples for this split
            if split == 'train':
                n = int(num_samples * 0.7)
            elif split == 'val':
                n = int(num_samples * 0.15)
            else:
                n = int(num_samples * 0.15)
            
            for i in tqdm(range(n), desc=f"{split}/{class_name}"):
                # Create synthetic image (random noise)
                img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                
                # Add some pattern to differentiate real from fake
                if class_name == 'fake':
                    # Add grid pattern for fake
                    img[::10, :] = 255
                    img[:, ::10] = 255
                
                # Save
                img_pil = Image.fromarray(img)
                img_pil.save(split_dir / f"{class_name}_{i:04d}.jpg")
    
    print("\n✓ Sample dataset created")
    analyze_dataset(output_dir)


def main():
    parser = argparse.ArgumentParser(description='Prepare deepfake detection dataset')
    parser.add_argument('--data_dir', type=str, default='../data',
                       help='Base data directory')
    parser.add_argument('--raw_dir', type=str, default='../data/raw',
                       help='Raw data directory')
    parser.add_argument('--action', type=str, required=True,
                       choices=['setup', 'preprocess', 'split', 'analyze', 'sample', 'info'],
                       help='Action to perform')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation set ratio')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples for sample dataset')
    
    args = parser.parse_args()
    
    if args.action == 'setup':
        # Create directory structure
        create_directory_structure(args.data_dir)
        print("\n✓ Setup complete!")
        print("\nNext steps:")
        print("1. Download datasets using --action info")
        print("2. Extract faces using --action preprocess")
        print("3. Split dataset using --action split")
        
    elif args.action == 'info':
        # Show download instructions
        download_sample_datasets()
        
    elif args.action == 'preprocess':
        # Extract faces from raw images
        print("\n" + "="*60)
        print("PREPROCESSING DATASET")
        print("="*60 + "\n")
        
        face_detector = FaceDetector()
        
        for class_name in ['real', 'fake']:
            input_dir = Path(args.raw_dir) / class_name
            output_dir = Path(args.data_dir) / 'processed' / class_name
            
            if input_dir.exists():
                print(f"\nProcessing {class_name} images...")
                extract_faces_from_images(input_dir, output_dir, face_detector)
            else:
                print(f"Warning: {input_dir} does not exist")
        
        print("\n✓ Preprocessing complete!")
        
    elif args.action == 'split':
        # Split dataset
        print("\n" + "="*60)
        print("SPLITTING DATASET")
        print("="*60 + "\n")
        
        processed_dir = Path(args.data_dir) / 'processed'
        
        split_dataset(
            processed_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=1 - args.train_ratio - args.val_ratio
        )
        
        print("\n✓ Dataset split complete!")
        
    elif args.action == 'analyze':
        # Analyze dataset
        analyze_dataset(args.data_dir)
        
    elif args.action == 'sample':
        # Create sample dataset
        create_sample_dataset(args.data_dir, args.num_samples)


if __name__ == "__main__":
    main()
