import os
import sys
import json
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.data.dataset import get_dataloaders
from src.models.image_baseline import DenseNet121Baseline
from src.evaluation.metrics import compute_classification_metrics

CONFIG_PATH = r"c:\Users\Subika M\OneDrive\Documents\final yr proj\pneumonia\configs\baseline_config.json"
SPLITS_CSV = r"c:\Users\Subika M\OneDrive\Documents\final yr proj\pneumonia\data\processed\splits.csv"
IMAGES_DIR = r"D:\rsna_pneumonia\images_224"
CHECKPOINT_DIR = r"c:\Users\Subika M\OneDrive\Documents\final yr proj\pneumonia\checkpoints"
RESULTS_DIR = r"c:\Users\Subika M\OneDrive\Documents\final yr proj\pneumonia\results"

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_epoch(model, loader, criterion, optimizer, device, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()
        
    running_loss = 0.0
    all_targets = []
    all_probs = []
    
    with torch.set_grad_enabled(is_train):
        for images, targets, _ in tqdm(loader, desc="Training" if is_train else "Evaluating", leave=False):
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)
            
            if is_train:
                optimizer.zero_grad()
                
            logits = model(images)
            loss = criterion(logits, targets)
            
            if is_train:
                loss.backward()
                optimizer.step()
                
            running_loss += loss.item() * images.size(0)
            
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.extend(probs.flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
            
    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_classification_metrics(all_targets, all_probs)
    metrics['loss'] = float(epoch_loss)
    return metrics, all_probs, all_targets

def plot_training_curves(history, save_path):
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    val_auc = [h['val_roc_auc'] for h in history]
    val_f1 = [h['val_f1_score'] for h in history]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curve
    axes[0].plot(epochs, train_loss, 'o-', label='Train Loss', color='blue')
    axes[0].plot(epochs, val_loss, 's--', label='Validation Loss', color='red')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (Weighted BCE)')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # AUC and F1
    axes[1].plot(epochs, val_auc, 'o-', label='Val ROC-AUC', color='green')
    axes[1].plot(epochs, val_f1, '^--', label='Val F1-Score', color='purple')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Validation Discrimination (ROC-AUC & F1)')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved training curves to: {save_path}")

def main():
    print("=" * 65)
    print("PHASE 5: IMAGE-ONLY DENSENET-121 TRANSFER LEARNING BASELINE")
    print("=" * 65)
    
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
        
    set_seed(config.get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using compute device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    # DataLoaders
    print(f"\nInitializing DataLoaders (Batch size: {config['batch_size']}, Workers: {config['num_workers']})...")
    train_loader, val_loader, test_loader = get_dataloaders(
        splits_csv=SPLITS_CSV,
        images_dir=IMAGES_DIR,
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)} | Test samples: {len(test_loader.dataset)}")
    
    # Loss function with positive class weighting
    pos_weight = torch.tensor([config['loss']['pos_weight']]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"Loss: BCEWithLogitsLoss with pos_weight={config['loss']['pos_weight']:.2f}")
    
    # Initialize Model: Stage 1 (Frozen Backbone)
    print("\n--- STAGE 1: FEATURE EXTRACTION (FROZEN BACKBONE) ---")
    model = DenseNet121Baseline(pretrained=config['pretrained'], freeze_backbone=True).to(device)
    param_summary = model.get_parameter_summary()
    print(f"Total parameters: {param_summary['total_parameters']:,}")
    print(f"Trainable parameters: {param_summary['trainable_parameters']:,} (Classifier head only)")
    print(f"Frozen parameters: {param_summary['frozen_parameters']:,}")
    
    stage1_cfg = config['stage1']
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=stage1_cfg['learning_rate'],
                      weight_decay=stage1_cfg['weight_decay'])
                      
    history = []
    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    early_stop_patience = config['early_stopping']['patience']
    
    best_checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_densenet121_baseline.pt")
    
    current_epoch = 1
    total_stage1 = stage1_cfg['epochs']
    
    for epoch in range(1, total_stage1 + 1):
        t0 = time.time()
        train_metrics, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        val_metrics, _, _ = run_epoch(model, val_loader, criterion, None, device, is_train=False)
        elapsed = time.time() - t0
        
        log_entry = {
            'epoch': current_epoch,
            'stage': 'Stage 1 (Feature Extraction)',
            'lr': stage1_cfg['learning_rate'],
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_sensitivity': val_metrics['sensitivity_recall'],
            'val_specificity': val_metrics['specificity'],
            'val_f1_score': val_metrics['f1_score'],
            'val_roc_auc': val_metrics['roc_auc'],
            'val_pr_auc': val_metrics['pr_auc'],
            'time_sec': elapsed
        }
        history.append(log_entry)
        
        print(f"Epoch [{current_epoch:02d}] (Stage 1) - Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
              f"Val ROC-AUC: {val_metrics['roc_auc']:.4f} | Val F1: {val_metrics['f1_score']:.4f} | "
              f"Sens: {val_metrics['sensitivity_recall']:.4f} | Spec: {val_metrics['specificity']:.4f} ({elapsed:.1f}s)")
              
        if val_metrics['roc_auc'] > best_val_auc:
            best_val_auc = val_metrics['roc_auc']
            best_epoch = current_epoch
            torch.save({
                'epoch': current_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }, best_checkpoint_path)
            print(f"  --> Saved new best checkpoint (ROC-AUC: {best_val_auc:.4f})")
            
        current_epoch += 1

    # Stage 2: Fine-Tuning Full Backbone
    print("\n--- STAGE 2: FINE-TUNING FULL BACKBONE ---")
    model.unfreeze_backbone()
    param_summary = model.get_parameter_summary()
    print(f"Trainable parameters: {param_summary['trainable_parameters']:,} (All layers unfrozen)")
    
    stage2_cfg = config['stage2']
    optimizer = AdamW(model.parameters(),
                      lr=stage2_cfg['learning_rate'],
                      weight_decay=stage2_cfg['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='max',
                                  factor=config['lr_scheduler']['factor'],
                                  patience=config['lr_scheduler']['patience'],
                                  min_lr=config['lr_scheduler']['min_lr'])
                                  
    total_stage2 = stage2_cfg['epochs']
    patience_counter = 0
    
    for epoch in range(1, total_stage2 + 1):
        t0 = time.time()
        train_metrics, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        val_metrics, _, _ = run_epoch(model, val_loader, criterion, None, device, is_train=False)
        elapsed = time.time() - t0
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_metrics['roc_auc'])
        
        log_entry = {
            'epoch': current_epoch,
            'stage': 'Stage 2 (Fine-Tuning)',
            'lr': current_lr,
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_sensitivity': val_metrics['sensitivity_recall'],
            'val_specificity': val_metrics['specificity'],
            'val_f1_score': val_metrics['f1_score'],
            'val_roc_auc': val_metrics['roc_auc'],
            'val_pr_auc': val_metrics['pr_auc'],
            'time_sec': elapsed
        }
        history.append(log_entry)
        
        print(f"Epoch [{current_epoch:02d}] (Stage 2) - Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
              f"Val ROC-AUC: {val_metrics['roc_auc']:.4f} | Val F1: {val_metrics['f1_score']:.4f} | "
              f"Sens: {val_metrics['sensitivity_recall']:.4f} | Spec: {val_metrics['specificity']:.4f} (lr={current_lr:.1e}, {elapsed:.1f}s)")
              
        if val_metrics['roc_auc'] > best_val_auc:
            best_val_auc = val_metrics['roc_auc']
            best_epoch = current_epoch
            patience_counter = 0
            torch.save({
                'epoch': current_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }, best_checkpoint_path)
            print(f"  --> Saved new best checkpoint (ROC-AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\n[Early Stopping Triggered] No improvement in Val ROC-AUC for {early_stop_patience} consecutive epochs.")
                break
                
        current_epoch += 1

    # Save training logs and curves
    log_df = pd.DataFrame(history)
    log_csv_path = os.path.join(RESULTS_DIR, "baseline_training_log.csv")
    log_df.to_csv(log_csv_path, index=False)
    print(f"\nSaved training history to: {log_csv_path}")
    
    curve_png_path = os.path.join(RESULTS_DIR, "baseline_training_curves.png")
    plot_training_curves(history, curve_png_path)
    
    # -------------------------------------------------------------------------
    # HELD-OUT TEST EVALUATION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("PHASE 8: EVALUATION ON HELD-OUT TEST SET (NEVER SEEN DURING TRAINING)")
    print("=" * 65)
    print(f"Loading best checkpoint from Epoch {best_epoch} ({best_checkpoint_path})...")
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics, test_probs, test_targets = run_epoch(model, test_loader, criterion, None, device, is_train=False)
    
    test_json_path = os.path.join(RESULTS_DIR, "baseline_test_metrics.json")
    with open(test_json_path, 'w') as f:
        json.dump(test_metrics, f, indent=4)
        
    val_json_path = os.path.join(RESULTS_DIR, "baseline_val_metrics.json")
    with open(val_json_path, 'w') as f:
        json.dump(checkpoint['val_metrics'], f, indent=4)
        
    print("\n--- HELD-OUT TEST PERFORMANCE METRICS ---")
    print(f"  Accuracy:            {test_metrics['accuracy'] * 100:.2f}%")
    print(f"  Sensitivity (Recall): {test_metrics['sensitivity_recall'] * 100:.2f}%")
    print(f"  Specificity:         {test_metrics['specificity'] * 100:.2f}%")
    print(f"  Precision (PPV):     {test_metrics['precision'] * 100:.2f}%")
    print(f"  F1-Score:            {test_metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:             {test_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:              {test_metrics['pr_auc']:.4f}")
    
    cm = test_metrics['confusion_matrix']
    print("\n--- CONFUSION MATRIX ---")
    print(f"  True Negatives (TN):  {cm['true_negatives']}")
    print(f"  False Positives (FP): {cm['false_positives']}")
    print(f"  False Negatives (FN): {cm['false_negatives']}")
    print(f"  True Positives (TP):  {cm['true_positives']}")
    print("=" * 65)

if __name__ == '__main__':
    main()
