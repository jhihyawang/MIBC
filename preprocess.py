import cv2
import os 
import pandas as pd
import random  
import numpy as np
# pip install opencv-python pandas numpy matplotlib
def crop_black_borders(img, threshold=5, margin_ratio=0.02):
    """
    img: numpy array, shape (H, W) 或 (H, W, 3)
    threshold: > threshold 視為「有訊號」
    margin_ratio: 在找到的 bounding box 外多保留的比例
    """
    # 如果是 3 channel，先轉灰階判斷 mask，但 crop 時保留原通道
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 建立「非背景」mask
    mask = gray > threshold

    # 如果整張圖都小於 threshold，就不 crop
    if not mask.any():
        return img

    ys, xs = np.where(mask)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    h, w = gray.shape
    margin_y = int(h * margin_ratio)
    margin_x = int(w * margin_ratio)

    y_min = max(y_min - margin_y, 0)
    y_max = min(y_max + margin_y, h - 1)
    x_min = max(x_min - margin_x, 0)
    x_max = min(x_max + margin_x, w - 1)

    # 注意 slicing 的 end index 要 +1
    if img.ndim == 3:
        cropped = img[y_min:y_max+1, x_min:x_max+1, :]
    else:
        cropped = img[y_min:y_max+1, x_min:x_max+1]

    return cropped

def all_flip_to_right(img, side='R'):
    """
    如果是左乳(R)，則水平翻轉成右乳(L)
    """
    if side == 'R':
        return cv2.flip(img, 1)  # 水平翻轉
    return img

def resize_padding_2_1(img, target_width=512):
    """
    Resize and pad image to 2:1 aspect ratio.
    - 垂直 padding 一律貼在「下方」
    - 水平 padding 一律貼在「右邊」
    """

    h, w = img.shape[:2]

    # 這裡如果你是想要 H:W = 2:1 的長圖，可以用：
    target_height = target_width * 2
    # 如果你要 W:H = 2:1 的橫圖，改成：
    # target_height = target_width // 2

    # 計算縮放比例
    scale = min(target_width / w, target_height / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    # resize
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # 建立黑底 canvas
    if resized.ndim == 3:
        channels = resized.shape[2]
        padded = np.zeros((target_height, target_width, channels), dtype=resized.dtype)
    else:
        padded = np.zeros((target_height, target_width), dtype=resized.dtype)

    # 🔸關鍵：貼在左上角 → padding 自然就跑到「下方 + 右邊」
    y_offset = 0
    x_offset = 0

    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return padded

def apply_clahe(img, clip_limit=1.5, tile_grid_size=(8, 8)):
    # Ensure the image is grayscale
    if img.ndim == 3:  # If the image has 3 channels (e.g., RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)

def normalize_for_calc(img, low=1):
    """
    只做低端剪裁，保留高亮細節
    """
    img_f = img.astype(np.float32)
    nonzero = img_f[img_f > 0]
    if nonzero.size < 10:
        return img

    p_low = np.percentile(nonzero, low)

    # 只剪低，不剪高
    img_f = np.clip(img_f, p_low, None)

    # 用實際 max 當上限，保留 highlight 尾端
    max_v = img_f.max()
    if max_v <= p_low:
        return img

    img_f = (img_f - p_low) / (max_v - p_low + 1e-6)
    img_f = np.clip(img_f * 255.0, 0, 255)
    return img_f.astype(np.uint8)

def process_image_pipeline(ori_img, save_root="Category 4"):
    # 讀取影像
    img = cv2.imread(ori_img)
    img_name = os.path.splitext(os.path.basename(ori_img))[0]
    folder_name = os.path.basename(os.path.dirname(ori_img))
    save_path = os.path.join(save_root, folder_name, img_name)
    print(f"儲存路徑: {save_path}")
    
    # Preprocessing steps
    cropped = crop_black_borders(img, threshold=5, margin_ratio=0.02)
    side = 'R' if 'R-' in os.path.basename(ori_img) else 'L'
    flipped = all_flip_to_right(cropped, side=side)
    resized_padded = resize_padding_2_1(flipped, target_width=512)
    normalized = normalize_for_calc(resized_padded, low=1.0)
    img_clahe = apply_clahe(normalized, clip_limit=0.7, tile_grid_size=(8, 8))

    # result = [resized_padded, img_clahe, normal_highpass]
    result = [img_clahe]
    for i, res in enumerate(result):
        suffix = ["processed_datasets"][i]
        print(f"儲存: {suffix}/{save_path}.jpg")
        new_save_path = os.path.join(suffix, save_root, folder_name, img_name)
        os.makedirs(os.path.dirname(new_save_path), exist_ok=True)
        cv2.imwrite(f"{new_save_path}.jpg", res)

def process_all(input_root="dataset"):
    for category in os.listdir(input_root):
        category_path = os.path.join(input_root, category)
        print(f"Processing category: {category}")
        if not os.path.isdir(category_path):
            continue
        for patient_id in os.listdir(category_path):
            patient_path = os.path.join(category_path, patient_id)
            if not os.path.isdir(patient_path):
                continue
            for img_file in os.listdir(patient_path):
                if img_file.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(patient_path, img_file)
                    try:
                        process_image_pipeline(img_path, save_root=os.path.join(category))
                    except Exception as e:
                        print(f"⚠️ 處理失敗: {img_path}, 錯誤: {e}")

### split dataset and generate csv ###
VALID_VIEWS = ['L-CC', 'R-CC', 'L-MLO', 'R-MLO']

def stratified_split(df, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42):
    """
    分層抽樣, 確保每個類別在 train/val/test 中的比例接近指定比例
    並且每個類別至少有 1 筆資料在每個集合（如果該類別總數 >= 3）
    這個版本使用四捨五入來計算每個集合的大小，以減少偏差。
    參數:
    - df: 包含 'label' 欄位的 DataFrame
    - train_ratio, val_ratio, test_ratio: 三個集合的比例和應為 1.0
    - seed: 隨機種子，確保可重現性
    回傳:
    - train_df, val_df, test_df: 分割後的 DataFrames
    """
    random.seed(seed)
    train_list, val_list, test_list = [], [], []

    for label in sorted(df['label'].unique()):
        df_label = df[df['label'] == label].sample(frac=1, random_state=seed)
        n_total = len(df_label)
        
        # 使用四捨五入計算各集合大小
        n_train = max(1, round(n_total * train_ratio))
        n_val = max(1, round(n_total * val_ratio))
        
        # 確保總數正確：test 集合吸收所有誤差
        n_test = max(1, n_total - n_train - n_val)
        
        # 處理邊界情況：總和超過 n_total
        if n_train + n_val + n_test > n_total:
            excess = n_train + n_val + n_test - n_total
            # 優先從最大的集合減少
            if n_train >= n_val and n_train > excess:
                n_train -= excess
            elif n_val > excess:
                n_val -= excess
            else:
                n_test -= excess
        
        # 處理邊界情況：總和小於 n_total（理論上不應發生）
        elif n_train + n_val + n_test < n_total:
            n_test += (n_total - (n_train + n_val + n_test))
        
        # 確保每個集合至少有 1 筆資料（如果類別總數 >= 3）
        if n_total >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
            n_test = max(1, n_test)
        
        train_list.append(df_label.iloc[:n_train])
        val_list.append(df_label.iloc[n_train:n_train+n_val])
        test_list.append(df_label.iloc[n_train+n_val:n_train+n_val+n_test])

    train_df = pd.concat(train_list).sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_list).sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_list).sample(frac=1, random_state=seed).reset_index(drop=True)

    return train_df, val_df, test_df


def generate_multiview_csvs(base_dir, output_dir,
                            train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42):
    random.seed(seed)
    patients_data = []

    print(f"\n📂 Scanning dataset folder: {base_dir}")
    
    for category in sorted(os.listdir(base_dir)):
        category_path = os.path.join(base_dir, category)
        if not os.path.isdir(category_path):
            continue
        try:
            label = int(category.replace("Category ", ""))
            if label == 6:
                print(f"⏭️ Skip Category {label}")
                continue
            elif label == 0:
                final_label = 0
            elif label==1 or label == 2 or label ==3:
                final_label = 1
            else:
                final_label = 2
        except ValueError:
            print(f"⚠️ Skip unrecognized folder name '{category}'")
            continue

        for patient_folder in os.listdir(category_path):
            patient_path = os.path.join(category_path, patient_folder)
            if not os.path.isdir(patient_path):
                continue

            patient_entry = {
                'patient_id': f"{category}/{patient_folder}",
                'label': final_label,
            }

            for img_file in os.listdir(patient_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    view_name = os.path.splitext(img_file)[0]
                    if view_name in VALID_VIEWS:
                        patient_entry[view_name] = os.path.join(category, patient_folder, img_file)

            patients_data.append(patient_entry)

    df = pd.DataFrame(patients_data)
    df = df.reindex(columns=VALID_VIEWS + ['label', 'patient_id'])

    missing_mask = df[VALID_VIEWS].isna().any(axis=1)
    if missing_mask.any():
        print("\n⚠️ Patients missing some views:")
        for _, row in df[missing_mask].iterrows():
            missing_views = [v for v in VALID_VIEWS if pd.isna(row[v])]
            print(f"  - {row['patient_id']} missing {', '.join(missing_views)}")

    df = df.dropna(subset=VALID_VIEWS)

    # 分層抽樣（使用改進版）
    train_df, val_df, test_df = stratified_split(df, train_ratio, val_ratio, test_ratio, seed)

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train_labels.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val_labels.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_labels.csv"), index=False)

    print(f"\n✅ Output complete: {len(df)} patients")
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"📁 Output folder: {output_dir}")

    # 每個類別統計（包含實際比例）
    print("\n📊 Dataset counts per class (stratified with rounding):")
    for label in sorted(df['label'].unique()):
        n_total_c = (df['label'] == label).sum()
        n_train_c = (train_df['label'] == label).sum()
        n_val_c   = (val_df['label'] == label).sum()
        n_test_c  = (test_df['label'] == label).sum()
        
        actual_train_ratio = n_train_c / n_total_c if n_total_c > 0 else 0
        actual_val_ratio = n_val_c / n_total_c if n_total_c > 0 else 0
        actual_test_ratio = n_test_c / n_total_c if n_total_c > 0 else 0
        
        print(f"  Category {label} (Total: {n_total_c}):")
        print(f"    Train={n_train_c} ({actual_train_ratio:.1%}), "
              f"Val={n_val_c} ({actual_val_ratio:.1%}), "
              f"Test={n_test_c} ({actual_test_ratio:.1%})")

if __name__ == "__main__":
    base_dir = "/media/stoneyew/512ssd/datasets"
    process_all(input_root=base_dir)
    output_dir = "csv/three_class"
    generate_multiview_csvs(
        base_dir=base_dir,
        output_dir=output_dir,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        seed=42
    )