"""
Evaluation Script for Deepfake Detection Models
Tests models on test set and generates performance metrics
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('../backend')

from models.ensemble import EnsembleDetector
from scripts.train_models import DeepfakeDataset, get_transforms


def evaluate_model(model, dataloader, device):
    """
    Evaluate model on test set
    
    Returns:
        predictions, labels, probabilities
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)
            
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).float()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probs)


def calculate_metrics(predictions, labels, probs):
    """Calculate evaluation metrics"""
    
    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions),
        'recall': recall_score(labels, predictions),
        'f1_score': f1_score(labels, predictions),
        'auc_roc': roc_auc_score(labels, probs)
    }
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)
    
    # Additional metrics
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return metrics


def plot_confusion_matrix(cm, save_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curve(labels, probs, save_path):
    """Plot ROC curve"""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_comparison(results, save_path):
    """Plot metrics comparison across models"""
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 4))
    
    for idx, metric in enumerate(metrics):
        values = [results[model][metric] * 100 for model in models]
        
        axes[idx].bar(models, values, color=['#667eea', '#764ba2', '#26de81', '#45aaf2'])
        axes[idx].set_ylabel('Score (%)')
        axes[idx].set_title(metric.replace('_', ' ').title())
        axes[idx].set_ylim([0, 105])
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(values):
            axes[idx].text(i, v + 2, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_ensemble(test_dir, batch_size=32):
    """Evaluate ensemble model"""
    print("\n" + "="*50)
    print("EVALUATING ENSEMBLE MODEL")
    print("="*50 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load test data
    test_dataset = DeepfakeDataset(test_dir, transform=get_transforms(train=False))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=4)
    
    print(f"Test dataset size: {len(test_dataset)} samples\n")
    
    # Initialize ensemble
    ensemble = EnsembleDetector(model_dir='../models', device=device)
    
    # Evaluate each model individually
    results = {}
    
    for model_name in ['cnn', 'efficientnet', 'xception', 'vit']:
        print(f"\nEvaluating {model_name.upper()}...")
        
        model = ensemble.get_model(model_name)
        predictions, labels, probs = evaluate_model(model, test_loader, device)
        
        metrics = calculate_metrics(predictions, labels, probs)
        results[model_name] = metrics
        
        print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"Precision: {metrics['precision']*100:.2f}%")
        print(f"Recall: {metrics['recall']*100:.2f}%")
        print(f"F1-Score: {metrics['f1_score']*100:.2f}%")
        print(f"AUC-ROC: {metrics['auc_roc']*100:.2f}%")
        
        # Save confusion matrix
        cm = confusion_matrix(labels, predictions)
        plot_confusion_matrix(cm, f'../docs/cm_{model_name}.png')
        
        # Save ROC curve
        plot_roc_curve(labels, probs, f'../docs/roc_{model_name}.png')
    
    # Evaluate ensemble predictions
    print("\n" + "-"*50)
    print("ENSEMBLE PREDICTIONS")
    print("-"*50 + "\n")
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Ensemble'):
            images_np = images.permute(0, 2, 3, 1).numpy()
            
            batch_preds = []
            batch_probs = []
            
            for img in images_np:
                pred, conf, _ = ensemble.predict(img)
                batch_preds.append(float(pred))
                batch_probs.append(conf)
            
            all_predictions.extend(batch_preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(batch_probs)
    
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    probs = np.array(all_probs)
    
    ensemble_metrics = calculate_metrics(predictions, labels, probs)
    results['ensemble'] = ensemble_metrics
    
    print(f"Accuracy: {ensemble_metrics['accuracy']*100:.2f}%")
    print(f"Precision: {ensemble_metrics['precision']*100:.2f}%")
    print(f"Recall: {ensemble_metrics['recall']*100:.2f}%")
    print(f"F1-Score: {ensemble_metrics['f1_score']*100:.2f}%")
    print(f"AUC-ROC: {ensemble_metrics['auc_roc']*100:.2f}%")
    
    # Save ensemble confusion matrix and ROC
    cm = confusion_matrix(labels, predictions)
    plot_confusion_matrix(cm, '../docs/cm_ensemble.png')
    plot_roc_curve(labels, probs, '../docs/roc_ensemble.png')
    
    # Plot comparison
    plot_metrics_comparison(results, '../docs/metrics_comparison.png')
    
    # Save results
    results_path = Path('../docs/evaluation_results.json')
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print("="*50)
    print(f"\nResults saved to: {results_path}")
    print(f"Visualizations saved to: ../docs/")
    
    # Print summary table
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)
    print(f"{'Model':<15} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'AUC':<8}")
    print("-"*50)
    for model, metrics in results.items():
        print(f"{model.upper():<15} "
              f"{metrics['accuracy']*100:>6.2f}% "
              f"{metrics['precision']*100:>6.2f}% "
              f"{metrics['recall']*100:>6.2f}% "
              f"{metrics['f1_score']*100:>6.2f}% "
              f"{metrics['auc_roc']*100:>6.2f}%")
    print("="*50 + "\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate deepfake detection models')
    parser.add_argument('--test_dir', type=str, default='../data/test',
                       help='Test data directory')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    
    args = parser.parse_args()
    
    evaluate_ensemble(args.test_dir, args.batch_size)


if __name__ == "__main__":
    main()
