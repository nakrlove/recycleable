import os
import shutil
import random
from pathlib import Path

def split_dataset(
    original_train_dir="dataset/train",
    split_dir="dataset_split",
    split_ratio=(0.8, 0.1, 0.1),
    max_per_class=None,      # None이면 전체 데이터 사용, 숫자면 그 갯수 기준
    seed=42
):
    random.seed(seed)

    original_train_dir = Path(original_train_dir)
    split_dir = Path(split_dir)

    # 새로운 폴더 구조 생성
    for folder in ["train", "val", "test"]:
        (split_dir / folder).mkdir(parents=True, exist_ok=True)

    # 클래스별 폴더 가져오기
    classes = [d.name for d in original_train_dir.iterdir() if d.is_dir()]
    class_counts = {cls: len(list((original_train_dir/cls).glob("*"))) for cls in classes}
    max_count = max(class_counts.values()) if max_per_class is None else max_per_class

    print("클래스별 원본 이미지 수:", class_counts)
    print("사용할 최대 클래스 수:", max_count)

    for cls in classes:
        cls_path = original_train_dir / cls
        images = list(cls_path.glob("*"))

        # max_per_class 적용: 소수 클래스는 반복, 대수 클래스는 랜덤 추출
        if len(images) < max_count:
            images = images + random.choices(images, k=max_count - len(images))
        elif len(images) > max_count:
            images = random.sample(images, max_count)

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

    print("데이터 분할 완료! '{}' 폴더에 train/val/test로 나누어 저장되었습니다.".format(split_dir))

# ================================
# 예시 사용
# ================================

# 1. 전체 데이터를 그대로 비율대로 나누기
#split_dataset(original_train_dir="dataset/train", split_dir="dataset_split_full")

# 2. 클래스별 파일 갯수를 1000개 기준으로 나누기
split_dataset(original_train_dir="dataset/train", split_dir="dataset_sp", max_per_class=100)
