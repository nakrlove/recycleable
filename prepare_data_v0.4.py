import os
import random
import shutil
from pathlib import Path

# ============================
# 설정
# ============================
SOURCE_DIR = "dataset/train"
TARGET_DIR = "dataset_sp100"
IMG_EXTENSION = ".jpg"

TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1

MODE = "fixed"      # "fixed" / "all"
FIXED_NUM = 100    # train 기준 (fixed 모드)
MIN_VAL_TEST = 20   # val/test 최소 샘플

# ============================
# 폴더 생성 함수
# ============================
def make_dirs(target_dir, class_names):
    for split in ["train", "val", "test"]:
        for cls in class_names:
            os.makedirs(os.path.join(target_dir, split, cls), exist_ok=True)

# ============================
# 데이터 split 함수
# ============================
def split_dataset(mode="all", fixed_num=5000, min_val_test=20):
    class_files = {}
    for cls in os.listdir(SOURCE_DIR):
        cls_path = os.path.join(SOURCE_DIR, cls)
        if os.path.isdir(cls_path):
            files = [os.path.join(cls_path, f) for f in os.listdir(cls_path)
                     if f.lower().endswith(IMG_EXTENSION)]
            class_files[cls] = files

    class_names = list(class_files.keys())
    make_dirs(TARGET_DIR, class_names)

    for cls, files in class_files.items():
        random.shuffle(files)
        total = len(files)

        if mode == "fixed":
            # train 기준
            train_count = min(fixed_num, total)
            remaining = total - train_count

            # val/test 최소 보장
            val_count = max(int(total * VAL_RATIO), min_val_test)
            test_count = max(int(total * TEST_RATIO), min_val_test)

            # 총합이 total보다 많으면 test→val 순으로 줄이고, train은 남은 샘플
            if train_count + val_count + test_count > total:
                excess = train_count + val_count + test_count - total
                reduce_test = min(excess, test_count - min_val_test)
                test_count -= reduce_test
                excess -= reduce_test

                reduce_val = min(excess, val_count - min_val_test)
                val_count -= reduce_val
                excess -= reduce_val

            train_count = total - val_count - test_count

            train_files = files[:train_count]
            val_files   = files[train_count:train_count+val_count]
            test_files  = files[train_count+val_count:train_count+val_count+test_count]

        elif mode == "all":
            # 기존 all 모드: 클래스별 전체 데이터 기준 비율 split
            train_count = int(total * TRAIN_RATIO)
            val_count   = int(total * VAL_RATIO)
            test_count  = total - train_count - val_count  # 남은 파일

            train_files = files[:train_count]
            val_files   = files[train_count:train_count+val_count]
            test_files  = files[train_count+val_count:]

        else:
            raise ValueError("mode는 'fixed' 또는 'all'만 가능합니다.")

        # 파일 복사
        for f in train_files:
            shutil.copy(f, os.path.join(TARGET_DIR, "train", cls))
        for f in val_files:
            shutil.copy(f, os.path.join(TARGET_DIR, "val", cls))
        for f in test_files:
            shutil.copy(f, os.path.join(TARGET_DIR, "test", cls))

    print(f"✅ Dataset split 완료! (mode={mode})")
    print(f"Saved to '{TARGET_DIR}' with train/val/test folders per class.")

# ============================
# 실행
# ============================
if __name__ == "__main__":
    split_dataset(mode=MODE, fixed_num=FIXED_NUM, min_val_test=MIN_VAL_TEST)
