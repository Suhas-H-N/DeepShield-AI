"""
Training Script for Deepfake Detection Models
Trains individual models and creates ensemble
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime

import sys
sys.path.append('../backend')

from models.cnn_model import DeepfakeCNN
from models.efficientnet import EfficientNetDetector
from models.xception import XceptionDetector
from models.vision_transformer import ViTDetector


class DeepfakeDataset(Dataset):
    """Custom dataset for deepfake detection"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Load image paths and labels
        self.samples = []
        
        # Real images (label 0)
        real_dir = self.data_dir / 'real'
        if real_dir.exists():
            for img_path in real_dir.glob('*.jpg'):
                self.samples.append((str(img_path), 0))
            for img_path in real_dir.glob('*.png'):
                self.samples.append((str(img_path), 0))
        
        # Fake images (label 1)
        fake_dir = self.data_dir / 'fake'
        if fake_dir.exists():
            for img_path in fake_dir.glob('*.jpg'):
                self.samples.append((str(img_path), 1))
            for img_path in fake_dir.glob('*.png'):
                self.samples.append((str(img_path), 1))
        
        print(f"Loaded {len(self.samples)} samples from {data_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        from PIL import Image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(train=True):
    """Get data transforms"""
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate accuracy
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': running_loss / (pbar.n + 1), 
                         'acc': 100 * correct / total})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation'):
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100 * correct / total
    
    return val_loss, val_acc


def train_model(model_name, train_dir, val_dir, epochs=50, batch_size=32, lr=0.001):
    """Train a specific model"""
    print(f"\n{'='*50}")
    print(f"Training {model_name.upper()}")
    print(f"{'='*50}\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = DeepfakeDataset(train_dir, transform=get_transforms(train=True))
    val_dataset = DeepfakeDataset(val_dir, transform=get_transforms(train=False))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=4)
    
    # Initialize model
    if model_name == 'cnn':
        model = DeepfakeCNN()
    elif model_name == 'efficientnet':
        model = EfficientNetDetector(version='b4')
    elif model_name == 'xception':
        model = XceptionDetector()
    elif model_name == 'vit':
        model = ViTDetector()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                           optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = Path('../models') / f'{model_name}_best.pth'
            save_path.parent.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, save_path)
            print(f"✓ Saved best model (acc: {val_acc:.2f}%)")
    
    # Save training history
    history_path = Path('../models') / f'{model_name}_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"{'='*50}\n")
    
    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description='Train deepfake detection models')
    parser.add_argument('--model', type=str, default='all',
                       choices=['cnn', 'efficientnet', 'xception', 'vit', 'all'],
                       help='Model to train')
    parser.add_argument('--train_dir', type=str, default='../data/train',
                       help='Training data directory')
    parser.add_argument('--val_dir', type=str, default='../data/val',
                       help='Validation data directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    models_to_train = ['cnn', 'efficientnet', 'xception', 'vit'] if args.model == 'all' else [args.model]
    
    results = {}
    
    for model_name in models_to_train:
        best_acc = train_model(
            model_name=model_name,
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )
        results[model_name] = best_acc
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    for model, acc in results.items():
        print(f"{model.upper()}: {acc:.2f}%")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
