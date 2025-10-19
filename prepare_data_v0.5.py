import os
import random
import shutil
from pathlib import Path
import numpy as np

# ============================
# 설정
# ============================
SOURCE_DIR = "dataset/train"
TARGET_DIR = "dataset_10000"
IMG_EXTENSION = ".jpg"

TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1

MODE = "fixed"      # "fixed" / "all"
FIXED_NUM = 10000    # train 기준 (fixed 모드)
MIN_VAL_TEST = 20   # val/test 최소 샘플

# 클래스 순서 정의 (중요!)
CLASS_ORDER = [
    "aluminum_can1", "aluminum_can2", "battery", "fluorescent_lamp",
    "glass_brown", "glass_clear", "glass_green", "paper1", "paper2",
    "pet_clear_single1", "pet_clear_single2", "pet_clear_single3",
    "pet_colored_single1", "pet_colored_single2", "pet_colored_single3",
    "plastic_pe1", "plastic_pe2",
    "plastic_pp1", "plastic_pp2", "plastic_pp3",
    "plastic_ps1", "plastic_ps2", "plastic_ps3",
    "steel_can1", "steel_can2", "steel_can3",
    "styrofoam1", "styrofoam2", "vinyl"
]

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

    # 클래스별 카운트 저장
    class_counts = []

    for cls in CLASS_ORDER:
        if cls not in class_files:
            print(f"⚠️ 클래스 '{cls}'가 SOURCE_DIR에 없습니다. 0으로 처리합니다.")
            class_counts.append(0)
            continue

        files = class_files[cls]
        random.shuffle(files)
        total = len(files)

        if mode == "fixed":
            # train 기준
            train_count = min(fixed_num, total)
            remaining = total - train_count

            # val/test 최소 보장
            val_count = max(int(total * VAL_RATIO), min_val_test)
            test_count = max(int(total * TEST_RATIO), min_val_test)

            # 총합이 total보다 많으면 test→val 순으로 줄이고 train은 남은 샘플
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
            # 기존 all 모드
            train_count = int(total * TRAIN_RATIO)
            val_count   = int(total * VAL_RATIO)
            test_count  = total - train_count - val_count

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

        # train 개수 기록
        class_counts.append(len(train_files))

    # ============================
    # CLASS_COUNTS 배열 로그 출력
    # ============================
    CLASS_COUNTS = np.array(class_counts, dtype=int)
    print("\n📊 CLASS_COUNTS 배열:")
    print("CLASS_COUNTS = np.array([")
    for i, count in enumerate(CLASS_COUNTS):
        end = "," if i < len(CLASS_COUNTS)-1 else ""
        print(f"    {count}{end}")
    print("])\n")

    print("✅ Dataset split 완료!")
    print(f"mode = {mode}")
    print(f"결과 저장 위치: {TARGET_DIR}")
    return CLASS_COUNTS


# ============================
# 실행
# ============================
if __name__ == "__main__":
    CLASS_COUNTS = split_dataset(mode=MODE, fixed_num=FIXED_NUM, min_val_test=MIN_VAL_TEST)
