import os
import shutil
import random
from pathlib import Path

# ================================
# 설정
# ================================
original_train_dir = Path("dataset/train")  # 기존 원본 train
split_dir = Path("dataset_split")           # 새로운 데이터 폴더
split_ratio = (0.8, 0.1, 0.1)              # train : val : test

# ================================
# 새로운 폴더 구조 생성
# ================================
for folder in ["train", "val", "test"]:
    (split_dir / folder).mkdir(parents=True, exist_ok=True)

# ================================
# 클래스별 이미지 수 확인
# ================================
classes = [d.name for d in original_train_dir.iterdir() if d.is_dir()]
class_counts = {cls: len(list((original_train_dir/cls).glob("*"))) for cls in classes}
max_count = max(class_counts.values())

print("클래스별 이미지 수:", class_counts)
print("최대 클래스 수:", max_count)

# ================================
# 분할 및 균형 맞추기
# ================================
for cls in classes:
    cls_path = original_train_dir / cls
    images = list(cls_path.glob("*"))
    
    # 클래스별 oversampling
    if len(images) < max_count:
        images = images + random.choices(images, k=max_count - len(images))
    
    random.shuffle(images)
    
    n_total = len(images)
    n_train = int(split_ratio[0] * n_total)
    n_val = int(split_ratio[1] * n_total)
    n_test = n_total - n_train - n_val
    
    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train+n_val]
    test_imgs = images[n_train+n_val:]
    
    # 각 폴더에 클래스별 서브폴더 생성 후 복사
    for folder, img_list in zip(["train", "val", "test"], [train_imgs, val_imgs, test_imgs]):
        cls_folder = split_dir / folder / cls
        cls_folder.mkdir(parents=True, exist_ok=True)
        for img_path in img_list:
            shutil.copy(img_path, cls_folder / img_path.name)

print("데이터 분할 완료! 'dataset_split' 폴더에 train/val/test로 나누어 저장되었습니다.")
