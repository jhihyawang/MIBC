import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm
import os
import argparse
from datetime import datetime

from dataloader import get_dataloaders
from utils import set_seed, plot_confusion_matrix, plot_training_curves, calculate_class_weights
from model import SiameseResNetRuleModel
import torch.nn.functional as F  

# ---------------- Exam Rule Loss ----------------
def exam_rule_loss(exam_log_prob, targets, class_weights=None):
    """
    exam_log_prob: (B,3) 來自 model 的 exam-level log 機率
    targets:   (B,)  exam-level label（0/1/2）
    class_weights: tensor(num_classes,) 或 None
    """
    # # 轉成 log prob
    # log_p = torch.log(exam_probs)

    if class_weights is not None:
        return F.nll_loss(exam_log_prob, targets, weight=class_weights)
    else:
        return F.nll_loss(exam_log_prob, targets)


# ==========================================
# 參數解析
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description='訓練 Mammography 多視角分類模型')
    
    # 資料集參數
    parser.add_argument('--csv_train', type=str, default='csv/six_classes/train_labels.csv',
                        help='訓練集 CSV 路徑')
    parser.add_argument('--csv_val', type=str, default='csv/six_classes/val_labels.csv',
                        help='驗證集 CSV 路徑')
    parser.add_argument('--csv_test', type=str, default='csv/six_classes/test_labels.csv',
                        help='測試集 CSV 路徑')
    parser.add_argument('--root_dir', type=str, default='datasets_v1',
                        help='圖片所在的根目錄')
    parser.add_argument('--img_height', type=int, default=1024,
                        help='影像高度')
    parser.add_argument('--img_width', type=int, default=512,
                        help='影像寬度')
    
    # 模型參數
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet18','resnet50', 'resnet101', 'resnet22_nyu', 'efficientnet_b0', 'efficientnet_b3', 
                                'efficientnet_b5', 'convnext_tiny', 'convnext_small', 'convnext_base'],
                        help='骨幹網路選擇')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='是否使用預訓練權重')
    parser.add_argument('--num_classes', type=int, default=6,
                        help='分類類別數量')
    parser.add_argument('--architecture', type=str, choices=['baseline','ipsi','bi','cross_view'], default='cross_view', help='模型架構')
    parser.add_argument('--concate_method', type=str, choices=['concat','concat_linear','concat_mlp'], default='concat', help='多視角特徵融合方式')
    parser.add_argument('--decision_rule', type=str, choices=['max','avg','rule'], default='max', help='exam-level 決策規則')

    # 訓練
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers 數量')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='訓練輪數')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='學習率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='權重衰減')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help='梯度累積步數')
    parser.add_argument('--mixed_precision', action='store_true', default=False,
                        help='是否使用混合精度訓練')
    parser.add_argument('--use_class_weights', action='store_true', default=False,
                        help='是否在損失函數中使用類別權重')
    parser.add_argument('--use_weighted_sampler', action='store_true', default=False,
                        help='是否使用加權隨機採樣器')
    # NYU 相關參數
    parser.add_argument('--nyu_weights_path', type=str, default=None,
                        help='NYU breast cancer classifier 預訓練權重路徑')
    # 其他參數
    parser.add_argument('--save_dir', type=str, default='experiments',
                        help='實驗根目錄')
    parser.add_argument('--experiment_id', type=str, default=None,
                        help='實驗 ID (若不指定則自動生成)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='運算設備')
    parser.add_argument('--eval_only', action='store_true',
                        help='僅進行測試集評估')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='載入的 checkpoint 路徑')
    parser.add_argument('--seed', type=int, default=42,
                        help='隨機種子')
    args = parser.parse_args()
    
    # 自動生成實驗 ID (根據主要配置)
    if args.experiment_id is None:
        timestamp = datetime.now().strftime("%m%d_%H%M")
        args.experiment_id = f"{args.backbone}_bs{args.batch_size}_lr{args.lr:.0e}_ep{args.num_epochs}_{timestamp}"
    
    # 設定裝置
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA 不可用，切換到 CPU")
        args.device = 'cpu'
    
    return args

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, scaler, args):
    model.train()
    running_loss = 0.0
    running_cls_loss = 0.0
    all_preds = []
    all_labels = []
    
    accumulation_steps = args.gradient_accumulation_steps
    
    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]")
    
    for batch_idx, (images, labels) in enumerate(loop):
        images = images.to(device)
        labels = labels.to(device)
        
        with torch.amp.autocast('cuda', enabled=args.mixed_precision):
            # ⭐ 新：model 回傳 exam_probs, left_logits, right_logits
            exam_log_prob, L_prob, R_prob, L_logits, R_logits = model(images)

            # 1. exam-level loss
            cls_loss = criterion(exam_log_prob, labels)

            loss = cls_loss 
            loss = loss / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        running_loss += loss.item() * accumulation_steps
        running_cls_loss += cls_loss.item()

        # ⭐ 用 exam_log_prob 取預測
        preds = torch.argmax(exam_log_prob, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        loop.set_postfix(
            loss=loss.item() * accumulation_steps,
            cls_loss=cls_loss.item(),
        )

    if len(loader) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
    epoch_loss = running_loss / len(loader)
    epoch_cls_loss = running_cls_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_cls_loss, epoch_acc
def validate(model, loader, criterion, device, args, phase="Valid"):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        loop = tqdm(loader, desc=f"[{phase}]")
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)
            
            with torch.amp.autocast('cuda', enabled=args.mixed_precision):
                exam_log_prob, L_prob, R_prob, L_logits, R_logits = model(images)
                loss = criterion(exam_log_prob, labels)
            
            running_loss += loss.item()
            
            preds = torch.argmax(exam_log_prob, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return epoch_loss, acc, f1, all_labels, all_preds

def test(model, loader, device, args, exp_dir):
    """測試集評估，輸出完整報告"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        loop = tqdm(loader, desc="[Test]")
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)
            
            with torch.amp.autocast('cuda', enabled=args.mixed_precision):
                exam_log_prob, L_prob, R_prob, L_logits, R_logits = model(images)
            
            preds = torch.argmax(exam_log_prob, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # 計算指標
    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, zero_division=0, digits=4)
    
    # 準備報告內容
    report_content = []
    report_content.append("="*60)
    report_content.append("📊 測試集最終結果")
    report_content.append("="*60)
    report_content.append(f"實驗 ID: {args.experiment_id}")
    report_content.append(f"骨幹網路: {args.backbone}")
    report_content.append(f"影像尺寸: {args.img_height}x{args.img_width}")
    report_content.append(f"Batch Size: {args.batch_size}")
    report_content.append(f"學習率: {args.lr}")
    report_content.append(f"訓練輪數: {args.num_epochs}")
    report_content.append("-"*60)
    report_content.append(f"Test Accuracy: {test_acc:.4f}")
    report_content.append(f"Test Macro-F1: {test_f1:.4f}")
    report_content.append("\n詳細分類報告:")
    report_content.append(report)
    report_content.append("\n混淆矩陣:")
    report_content.append(str(cm))
    report_content.append("="*60)
    
    # 輸出到終端
    for line in report_content:
        print(line)
    
    # 儲存報告到檔案
    report_path = 'report3.txt'
    with open(report_path, 'a+', encoding='utf-8') as f:
        f.write('\n'.join(report_content))
    print(f"\n✅ 測試報告已儲存至: {report_path}")
    
    cm_path = os.path.join(exp_dir, f'cm_{args.experiment_id}.png')
    plot_confusion_matrix(all_labels, all_preds, cm_path, args.num_classes, phase=f"{args.experiment_id}")
    
    return test_acc, test_f1, all_labels, all_preds

def main():
    args = parse_args()
    set_seed(args.seed)
    # 建立實驗目錄結構
    exp_dir = os.path.join(args.save_dir, args.experiment_id)
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("🏥 Mammography 多視角分類訓練系統")
    print("="*60)
    print(f"實驗 ID: {args.experiment_id}")
    print(f"實驗目錄: {exp_dir}")
    print(f"骨幹網路: {args.backbone}")
    print(f"影像尺寸: {args.img_height}x{args.img_width}")
    print(f"Batch Size: {args.batch_size} (有效: {args.batch_size * args.gradient_accumulation_steps})")
    print(f"訓練輪數: {args.num_epochs}")
    print(f"學習率: {args.lr}")
    print(f"設備: {args.device}")
    print(f"混合精度: {args.mixed_precision}")
    print(f"隨機種子: {args.seed}")
    print(f"num_workers: {args.num_workers}")
    print("="*60 + "\n")
    
    device = torch.device(args.device)
    
    # 儲存實驗配置
    config_path = os.path.join(exp_dir, 'config.txt')
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write("實驗配置\n")
        f.write("="*60 + "\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
    print(f"✅ 實驗配置已儲存至: {config_path}\n")
    
    # 1. 計算權重 (這一步解決您的不平衡問題)
    if os.path.exists(args.csv_train) and args.use_class_weights:
        class_weights = calculate_class_weights(args.csv_train, args.num_classes, device)
    else:
        print("no class weights used in CE.")

    # 2. 準備 DataLoader
    img_size = (args.img_height, args.img_width)
    train_loader, val_loader, test_loader = get_dataloaders(
        csv_path_train=args.csv_train,
        csv_path_val=args.csv_val,
        csv_path_test=args.csv_test,
        root_dir=args.root_dir,
        img_size=img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        use_weighted_sampler=args.use_weighted_sampler
    )
    
    # 3. 初始化模型
    model = SiameseResNetRuleModel(
        backbone_name=args.backbone, 
        pretrained=args.pretrained, 
        num_classes=args.num_classes,
        architecture=args.architecture,
        concate_method=args.concate_method,
        decision_rule=args.decision_rule
    )
    model = model.to(device)
    
    # 載入 NYU 預訓練權重 (如果使用 resnet22_nyu)
    if args.backbone == 'resnet22_nyu' and args.nyu_weights_path:
        if os.path.exists(args.nyu_weights_path):
            model.load_nyu_pretrained(args.nyu_weights_path)
        else:
            print(f"⚠️  NYU 權重檔案不存在: {args.nyu_weights_path}")
            print("   將使用隨機初始化的權重")
    elif args.backbone == 'resnet22_nyu':
        print("⚠️  使用 resnet22_nyu 但未指定 --nyu_weights_path，將使用隨機初始化")
    
    # 載入 checkpoint (如果有)
    if args.checkpoint:
        print(f"載入 checkpoint: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print("✅ Checkpoint 載入成功\n")
    
    # 如果只是評估模式
    if args.eval_only:
        if test_loader is None:
            print("❌ 評估模式需要測試集")
            return
        if args.checkpoint is None:
            print("❌ 評估模式需要指定 checkpoint")
            return
        test(model, test_loader, device, args, exp_dir)
        return
    
    # 4. Loss Function for exam-rule
    if args.use_class_weights:
        print("使用類別權重於損失函數中 (exam rule loss)")
        # 建立一個 closure，把 class_weights 固定住
        def criterion(exam_probs, targets):
            return exam_rule_loss(exam_probs, targets, class_weights)
    else:
        def criterion(exam_probs, targets):
            return exam_rule_loss(exam_probs, targets, None)

    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # 混合精度訓練 (AMP) - 節省記憶體並加速
    scaler = torch.amp.GradScaler('cuda', enabled=args.mixed_precision)
    
    best_f1 = 0.0
    
    # 訓練歷史記錄
    history = {
        'train_loss': [],
        'train_cls_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    
    # 5. 訓練循環
    for epoch in range(args.num_epochs):
        # Train
        train_loss, train_cls_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scaler, args
        )
        
        # Valid
        val_loss, val_acc, val_f1, val_labels, val_preds = validate(
            model, val_loader, criterion, device, args, phase="Valid"
        )
        
        scheduler.step()
        
        # 記錄歷史
        history['train_loss'].append(train_loss)
        history['train_cls_loss'].append(train_cls_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"\nEpoch {epoch+1}/{args.num_epochs} Stats:")
        print(f"Train Loss: {train_loss:.4f} (Cls: {train_cls_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Macro-F1: {val_f1:.4f}")
        
        # 每 5 個 epoch 印出詳細報告並繪製驗證集混淆矩陣
        if (epoch + 1) % 5 == 0:
            print("\nClassification Report:")
            print(classification_report(val_labels, val_preds, zero_division=0))
            
        # 儲存最佳模型 (以 Macro F1 為準，比較能反映少數類別的表現)
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_path = os.path.join(checkpoint_dir, f"best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"🔥 New Best Model Saved! (F1: {best_f1:.4f})")
            
    # 繪製訓練曲線
    plot_training_curves(history, save_dir=exp_dir, title=args.experiment_id)

    # 6. 訓練結束後，在測試集上評估
    if test_loader is not None:
        print("\n" + "="*60)
        print("🎯 開始在測試集上評估最佳模型...")
        print("="*60)
        # 載入最佳模型
        best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
        model.load_state_dict(torch.load(best_model_path))
        test(model, test_loader, device, args, exp_dir)

if __name__ == "__main__":
    main()