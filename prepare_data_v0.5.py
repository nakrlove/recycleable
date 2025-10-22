import os
import random
import shutil
import numpy as np

# ============================
# 설정
# ============================
SOURCE_DIR = "dataset/train"
TARGET_DIR = "dataset_s100"
IMG_EXTENSION = ".jpg"

TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1

MODE = "fixed"      # "fixed" / "all"
FIXED_NUM = 100     # train 기준 (fixed 모드)
MIN_VAL_TEST = 20   # val/test 최소 샘플

# 클래스 순서 정의
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
# 폴더 생성
# ============================
def make_dirs(target_dir, class_names):
    for split in ["train", "val", "test"]:
        for cls in class_names:
            os.makedirs(os.path.join(target_dir, split, cls), exist_ok=True)

# ============================
# 데이터 split
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

    class_counts = []

    for cls in CLASS_ORDER:
        if cls not in class_files:
            print(f"⚠️ 클래스 '{cls}' 없음. 0으로 처리.")
            class_counts.append(0)
            continue

        files = class_files[cls]
        random.shuffle(files)
        total = len(files)

        if total == 0:
            print(f"⚠️ '{cls}' 클래스에 데이터 없음.")
            class_counts.append(0)
            continue

        # ========================================
        # 🔹 fixed 모드
        # ========================================
        if mode == "fixed":
            # train 기준 개수 설정
            train_count = min(fixed_num, total)

            # 남은 샘플 중 val/test 개수 계산
            val_count = int(train_count * (VAL_RATIO / TRAIN_RATIO))
            test_count = int(train_count * (TEST_RATIO / TRAIN_RATIO))

            # 최소 보장
            if val_count == 0:
                val_count = min(min_val_test, total - train_count)
            if test_count == 0:
                test_count = min(min_val_test, total - train_count - val_count)

            # 총합 초과 시 조정
            if train_count + val_count + test_count > total:
                overflow = train_count + val_count + test_count - total
                if test_count > min_val_test:
                    reduce = min(overflow, test_count - min_val_test)
                    test_count -= reduce
                    overflow -= reduce
                if overflow > 0 and val_count > min_val_test:
                    reduce = min(overflow, val_count - min_val_test)
                    val_count -= reduce
                    overflow -= reduce
                train_count = total - val_count - test_count

        # ========================================
        # 🔹 all 모드
        # ========================================
        elif mode == "all":
            train_count = int(total * TRAIN_RATIO)
            val_count   = int(total * VAL_RATIO)
            test_count  = total - train_count - val_count

            # val/test가 0이면 최소 보장
            if val_count == 0:
                val_count = min(min_val_test, total - train_count)
            if test_count == 0:
                test_count = min(min_val_test, total - train_count - val_count)

            # 총합이 초과되면 비율 유지하며 조정
            if train_count + val_count + test_count > total:
                excess = train_count + val_count + test_count - total
                train_count -= excess  # train 우선 줄이기
                if train_count < 0:
                    train_count = 0

        else:
            raise ValueError("mode는 'fixed' 또는 'all'만 가능합니다.")

        # ========================================
        # 파일 분배 (중복 없이)
        # ========================================
        train_files = files[:train_count]
        val_files   = files[train_count:train_count+val_count]
        test_files  = files[train_count+val_count:train_count+val_count+test_count]

        # 파일 복사
        for f in train_files:
            shutil.copy(f, os.path.join(TARGET_DIR, "train", cls))
        for f in val_files:
            shutil.copy(f, os.path.join(TARGET_DIR, "val", cls))
        for f in test_files:
            shutil.copy(f, os.path.join(TARGET_DIR, "test", cls))

        # 로그 출력
        print(f"✅ {cls:25s} | train={len(train_files):4d} | val={len(val_files):4d} | test={len(test_files):4d}")
        class_counts.append(len(train_files))

    # ============================
    # CLASS_COUNTS 배열 로그 출력
    # ============================
    CLASS_COUNTS = np.array(class_counts, dtype=int)
    print("\n📊 CLASS_COUNTS = np.array([")
    for i, c in enumerate(CLASS_COUNTS):
        end = "," if i < len(CLASS_COUNTS)-1 else ""
        print(f"    {c}{end}")
    print("])\n")
    print(f"✅ Split 완료 ({mode} 모드, 저장경로: {TARGET_DIR})")

    return CLASS_COUNTS

# ============================
# 실행
# ============================
if __name__ == "__main__":
    CLASS_COUNTS = split_dataset(mode=MODE, fixed_num=FIXED_NUM, min_val_test=MIN_VAL_TEST)
