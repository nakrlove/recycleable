"""
재활용 가능한 쓰레기를 찾는 딥러닝을 구현 되어있습니다.
하지만 학습할 데이터준비가 미비해서 프로그램으로 구현이 필요합니다

조건은
재활용가능한 이미지 dataset/train만 있는상태입니다. 
train데이터를 val,test폴드를 만들고 그곳에 비율대로 나누어 담을수  있는 방법과 로직을 구현해주세요
비율은 8:1:1입니다.
아래 train서브 폴드에 담긴 이미지 개수를 정리했습니다.
steel_can1.                : 13020
steel_can2                : 28160
steel_can3.               : 1301
aluminum_can1.       : 26199
aluminum_can2.      : 13532
paper1                       : 24578
paper2                       : 15126
pet_clear_single1.     : 23831
pet_clear_single2.     : 24845
pet_clear_single3.     : 24155
pet_colored_single1   : 27144 
pet_colored_single2.  : 26781
pet_colored_single3.  : 7726
plastic_pe1                  : 29493
plastic_pe2                 : 18249
plastic_pp1                  :25135
plastic_pp2                 :28616 
plastic_pp3                 : 19187
plastic_ps1                  : 30095
plastic_ps2                  : 29209
plastic_ps3                 : 13577
styrofoam1                   : 25884
styrofoam2                  : 2129
vinyl                              :16820
glass_brown               : 16807
glass_green                 : 16777
glass_clear                   :17125
battery                          :3273
fluorescent_lamp.       : 2404

현재 dataset/train 폴더에 각 클래스별 이미지가 있음.
1) 
train은 기존 train 폴더와 데이터를 그대로 두고
2) train 데이터 개수를 임의로 주면 임의의 수로만 채워주세요

아래는 공통내용입니다.
val, test 폴더를 새로 만들고 이미지들을 비율대로 나눠야 함.
클래스별 갯수가 일정하지 않으므로, 최대 클래스 수에 맞춰서 균형 조정 (oversampling) 가능하면 적용.
원본 dataset 데이터폴드는 그대로 두고 새로운 dataset_split 폴드를 만들고 그곳에 나누어 담아 주세요

"""
import os
import random
import shutil
from pathlib import Path

# ============================
# 설정
# ============================
SOURCE_DIR = "dataset/train"
TARGET_DIR = "dataset_sp500"
IMG_EXTENSION = ".jpg"

TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1

# mode = "all" / "fixed"
MODE = "fixed"  # "all"은 전체 split, "fixed"는 train 기준 split
FIXED_NUM = 1500  # mode="fixed"일 때 train 파일 개수 기준

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
def split_dataset(mode="all", fixed_num=5000):
    # 1️⃣ 클래스별 파일 수집
    class_files = {}
    for cls in os.listdir(SOURCE_DIR):
        cls_path = os.path.join(SOURCE_DIR, cls)
        if os.path.isdir(cls_path):
            files = [os.path.join(cls_path, f) for f in os.listdir(cls_path)
                     if f.lower().endswith(IMG_EXTENSION)]
            class_files[cls] = files

    class_names = list(class_files.keys())
    make_dirs(TARGET_DIR, class_names)

    # 2️⃣ 클래스별 split
    for cls, files in class_files.items():
        random.shuffle(files)
        total = len(files)

        if mode == "fixed":
            # train 파일 수를 기준으로 전체 8:1:1 비율 역산
            train_count = fixed_num
            total_count = int(train_count / TRAIN_RATIO)
            val_count   = int(total_count * VAL_RATIO)
            test_count  = total_count - train_count - val_count  # 남은 수

            if total_count > total:
                print(f"⚠️ 클래스 {cls}: 요청한 FIXED_NUM 대비 파일 부족 ({total_count} > {total})")
                total_count = total
                train_count = min(train_count, total_count)
                val_count   = int(total_count * VAL_RATIO)
                test_count  = total_count - train_count - val_count

            # 실제 파일 선택
            train_files = files[:train_count]
            val_files   = files[train_count:train_count+val_count]
            test_files  = files[train_count+val_count:train_count+val_count+test_count]

        else:  # mode="all"
            train_count = int(total * TRAIN_RATIO)
            val_count   = int(total * VAL_RATIO)
            test_count  = total - train_count - val_count
            train_files = files[:train_count]
            val_files   = files[train_count:train_count+val_count]
            test_files  = files[train_count+val_count:]

        # 3️⃣ 파일 복사
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
    split_dataset(mode=MODE, fixed_num=FIXED_NUM)
