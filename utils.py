from typing import Dict, Tuple
import cv2
import numpy as np
import pandas as pd
import torch
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def set_seed(seed):
    """
    設置隨機種子以確保實驗結果可重現。
    """
    random.seed(seed)  # 設置 Python 的內建隨機數生成器
    np.random.seed(seed)  # 設置 NumPy 的隨機數生成器
    torch.manual_seed(seed)  # 設置 PyTorch 的隨機數生成器（CPU）
    torch.cuda.manual_seed(seed)  # 設置 PyTorch 的隨機數生成器（GPU）
    torch.cuda.manual_seed_all(seed)  # 如果有多個 GPU，為所有 GPU 設置隨機數生成器
    torch.backends.cudnn.deterministic = True  # 確保 CUDA 的操作是確定性的
    torch.backends.cudnn.benchmark = False  # 禁用自動優化

def plot_confusion_matrix(y_true, y_pred, save_path, num_classes=6, phase="Validation"):
    """繪製並儲存混淆矩陣"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(num_classes), 
                yticklabels=range(num_classes))
    plt.title(f'Confusion Matrix - {phase}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 混淆矩陣已儲存至: {save_path}")

def plot_training_curves(history, save_dir, title=None):
    """繪製訓練曲線 (loss, acc, f1)"""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)

    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # F1 Score
    axes[2].plot(epochs, history['train_f1'], 'b-', label='Train Macro-F1', linewidth=2)
    axes[2].plot(epochs, history['val_f1'], 'g-', label='Val Macro-F1', linewidth=2)
    axes[2].set_title('Training and Validation Macro-F1 Score', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 訓練曲線已儲存至: {save_path}")

def calculate_class_weights(csv_path, num_classes, device):
    """
    根據訓練資料計算 Class Weights (處理不平衡)
    公式: Weight_c = Total_Samples / (Num_Classes * Count_c)
    """
    df = pd.read_csv(csv_path)
    labels = df['label'].values
    # 計算每個類別的數量
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    # 避免數量為 0 導致除以零 (加上 epsilon)
    raw = total / (num_classes * (counts + 1e-6))
    raw_mean = raw.mean()
    weights = raw / raw_mean
    # weights[0] *= 1.2
    # weights[2] *= 1.5
    print("\n=== 類別分佈與權重 ===")
    for i, (count, w) in enumerate(zip(counts, weights)):
        print(f"Class {i}: {count} 筆 -> Weight: {w:.4f}")
    print("======================\n")

    return torch.tensor(weights, dtype=torch.float32).to(device)