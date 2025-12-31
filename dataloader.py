import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import random
from typing import Tuple

# 防止 OpenCV 多執行緒與 PyTorch DataLoader 衝突
cv2.setNumThreads(0)

# --- 設定常數 ---
VIEWS = ["L-CC", "R-CC", "L-MLO", "R-MLO"]
# 使用 ImageNet 統計數據，因為通常會用預訓練模型
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def seed_worker(worker_id):
    """保證多進程 DataLoader 的 Augmentation 行為可復現"""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class MammoDataset(Dataset):
    def __init__(self, 
                 csv_path: str, 
                 root_dir: str = None,
                 img_size: Tuple[int, int] = (1024, 512),
                 train: bool = True,
                 enable_lr_swap: bool = True, 
                 ensure_orientation: bool = False,
                 seed: int = 42):
        
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir or os.path.dirname(csv_path)
        self.train = train
        self.height, self.width = img_size
        self.enable_lr_swap = bool(enable_lr_swap)
        self.ensure_orientation = bool(ensure_orientation)
        self.seed = int(seed)

        # 檢查必要欄位
        required = set(VIEWS + ["label"])
        if not required.issubset(self.df.columns):
            raise ValueError(f"CSV 缺少欄位: {required - set(self.df.columns)}")

        # 建立增強管線
        self.transforms = self.build_transforms(seed=self.seed)

    def build_transforms(self, seed: int = None) -> A.Compose:
        """
        建立強化的影像處理管線 (修正版 v2 - 針對新版 GaussNoise)
        """
        aug_list = [
            # 1. 基礎 Resize (使用 Cubic 插值保留細節)
            A.Resize(height=self.height, width=self.width, interpolation=cv2.INTER_CUBIC)
        ]
        
        if self.train:
            aug_list += [
                # ---幾何變換---
                # A.RandomResizedCrop(
                #         size=(self.height, self.width),  # 修改：使用 size 而非 height/width
                #         scale=(0.8, 1.0), 
                #         ratio=(0.75, 1.33), 
                #         p=0.5
                #     ),

                # [新增 2]: 翻轉 (乳房左右翻轉不影響病灶性質)
                A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.5), # 視情況開啟，有些乳房攝影上下翻轉會不自然

                # [原本的幾何變換]: 稍微調高旋轉角度
                A.OneOf([
                    # ElasticTransform: 保持你原本的設定，這對醫學影像很好
                    A.ElasticTransform(
                        alpha=120, 
                        sigma=120 * 0.05, 
                        p=0.7
                    ),
                    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
                    
                    # 加大旋轉角度到 45 度，增加難度
                    A.Affine(
                        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # 稍微增加平移
                        scale=(0.8, 1.2), # 稍微增加縮放範圍
                        rotate=(-45, 45), # [修改]: 15 -> 45
                        p=0.5
                    ),
                ], p=0.6),

                # [像素級變換]
                A.OneOf([
                    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5), # 稍微增加對比度變化
                    
                    # [新增 3]: 模糊 (模擬 BI-RADS 0 的不確定性)
                    # A.GaussianBlur(blur_limit=(3, 7), p=0.2),
                    
                    A.GaussNoise(
                        std_range=(0.02, 0.1), 
                        mean_range=(0.0, 0.0), 
                        p=0.3
                    ),
                ], p=0.5),

                # [正則化]
                # CoarseDropout: 保持你原本的設定，這是好的
                # A.CoarseDropout(
                #     num_holes_range=(1, 8),
                #     hole_height_range=(int(self.height*0.05), int(self.height*0.1)),
                #     hole_width_range=(int(self.width*0.05), int(self.width*0.1)),
                #     p=0.3
                # ),
            ]

        # 4. 正規化與轉 Tensor
        aug_list += [
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0), 
            ToTensorV2()
        ]

        return A.Compose(
            aug_list,
            additional_targets={
                "R-CC": "image",
                "L-MLO": "image",
                "R-MLO": "image",
            },
            is_check_shapes=False,
            seed=seed 
        )

    def _read_img(self, rel_path: str) -> np.ndarray:
        full_path = os.path.join(self.root_dir, rel_path)
        if not os.path.exists(full_path):
             # 實務上建議記錄 log 並跳過，這裡簡化為報錯
            raise FileNotFoundError(f"Image not found: {full_path}")

        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"OpenCV failed to read: {full_path}")

        # 轉為 3 Channel (Stack)
        img_3c = np.stack([img, img, img], axis=-1) 
        return img_3c

    @staticmethod
    def _hflip(img: np.ndarray) -> np.ndarray:
        return cv2.flip(img, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # 1. 讀取
        img_dict = {v: self._read_img(row[v]) for v in VIEWS}

        # 2. 方向校正 (統一乳頭朝右)
        if self.ensure_orientation:
            img_dict["R-CC"]  = self._hflip(img_dict["R-CC"])
            img_dict["R-MLO"] = self._hflip(img_dict["R-MLO"])

        # 3. 同步增強
        transformed = self.transforms(
            image=img_dict["L-CC"],
            **{"R-CC": img_dict["R-CC"], "L-MLO": img_dict["L-MLO"], "R-MLO": img_dict["R-MLO"]}
        )

        tensors = {
            "L-CC": transformed["image"],
            "R-CC": transformed["R-CC"],
            "L-MLO": transformed["L-MLO"],
            "R-MLO": transformed["R-MLO"]
        }

        # 4. [Optional] 隨機左右交換 (資料量少時建議開啟)
        # 交換 L/R 代表病患鏡像翻轉，對模型判讀無影響但能增加多樣性
        if self.train and self.enable_lr_swap and random.random() < 0.5:
            tensors["L-CC"],  tensors["R-CC"]  = tensors["R-CC"],  tensors["L-CC"]
            tensors["L-MLO"], tensors["R-MLO"] = tensors["R-MLO"], tensors["L-MLO"]

        # 5. Stack -> (4, 3, H, W)
        stacked_imgs = torch.stack([
            tensors["L-CC"], tensors["R-CC"], tensors["L-MLO"], tensors["R-MLO"]
        ], dim=0)

        label = torch.tensor(int(row["label"]), dtype=torch.long)

        return stacked_imgs, label

def get_dataloaders(
    csv_path_train: str, 
    csv_path_val: str, 
    csv_path_test: str,
    root_dir: str,
    img_size: Tuple[int, int] = (1024, 512), 
    batch_size: int = 8,
    num_workers: int = 4,
    seed: int = 42,
    use_weighted_sampler: bool = False
):
    g = torch.Generator()
    g.manual_seed(seed)

    # --- Train ---
    train_ds = MammoDataset(
        csv_path=csv_path_train, root_dir=root_dir, img_size=img_size,
        train=True, ensure_orientation=False, seed=seed
    )

    shuffle = True
    sampler = None

    # 針對資料不平衡的處理
    if use_weighted_sampler:
        print("⚖️  啟用 WeightedRandomSampler 對抗資料不平衡...")
        targets = train_ds.df["label"].values
        class_counts = np.bincount(targets)
        
        # 顯示當前類別分佈
        print(f"   類別分佈: {dict(enumerate(class_counts))}")
        
        # 計算權重 (數量越少權重越大)
        class_weights = 1. / (class_counts + 1e-6)
        samples_weights = class_weights[targets]
        
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(samples_weights).float(),
            num_samples=len(samples_weights),
            replacement=True
        )
        shuffle = False # 有 Sampler 時不可 shuffle

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, 
        shuffle=shuffle, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        worker_init_fn=seed_worker, generator=g
    )

    # --- Val ---
    val_ds = MammoDataset(
        csv_path=csv_path_val, root_dir=root_dir, img_size=img_size,
        train=False, seed=seed
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        worker_init_fn=seed_worker, generator=g
    )

    # --- Test ---
    test_ds = MammoDataset(
        csv_path=csv_path_test, root_dir=root_dir, img_size=img_size,
        train=False, seed=seed
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        worker_init_fn=seed_worker, generator=g
    )

    return train_loader, val_loader, test_loader