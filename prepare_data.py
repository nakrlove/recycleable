import os
import shutil
import splitfolders
import random
def prepare_and_split_data(input_dir="dataset/train", output_dir="split_dataset", ratio=(0.8, 0.2, 0.0)):
    """
    원본 데이터를 학습/검증 데이터셋으로 분리하고, 필요한 폴더를 확인/생성합니다.
    """
    
    # 1. 입력 폴더 확인: 원본 이미지가 있어야 할 경로
    if not os.path.exists(input_dir) or not os.listdir(input_dir):
        print(f"\n❌ 오류: 원본 입력 폴더 '{input_dir}'가 존재하지 않거나, 그 안에 클래스별 하위 폴더 및 이미지가 없습니다.")
        if not os.path.exists(input_dir):
            os.makedirs(input_dir, exist_ok=True)
            print(f"   👉 '{input_dir}' 폴더를 생성했습니다. 이미지를 넣은 후 다시 실행해 주세요.")
        return None 
    
    # 2. 출력 폴더 확인: 이미 분리된 데이터셋이 존재하는지 확인
    if os.path.exists(output_dir) and os.path.exists(os.path.join(output_dir, "train")) and os.path.exists(os.path.join(output_dir, "val")):
        print(f"\n✅ 분리된 데이터셋 폴더 '{output_dir}'가 이미 존재합니다. 분리 과정을 건너뛰고 기존 데이터를 사용합니다.")
        return output_dir # 성공적으로 경로 문자열 반환
    
    # 3. 분리 작업 수행
    print(f"\n✨ 분리된 데이터셋 폴더 '{output_dir}'가 없어 새로 분리 작업을 시작합니다.")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    try:
        splitfolders.ratio(
            input_dir, 
            output=output_dir, 
            seed=42, 
            ratio=ratio, 
            group_prefix=None, 
            move=False
        )
        print("✨ 데이터 분리 완료!")
        print(f"   -> 학습 데이터: {output_dir}/train")
        print(f"   -> 검증 데이터: {output_dir}/val")
        
        # 🚨 핵심 수정: 성공 시 반드시 경로를 반환합니다.
        return output_dir
        
    except FileNotFoundError:
        print(f"\n❌ 오류: 원본 입력 폴더 '{input_dir}'가 존재하지 않거나, 그 안에 클래스별 하위 폴더 및 이미지가 없습니다.")
        return None 
    except Exception as e:
        print(f"❌ 데이터 분리 중 치명적인 오류 발생: {e}")
        return None
    

import os
import shutil
import random
import math

def split_folder_fixed_count():
    # ----------------- 설정 -----------------
    ORIGINAL_DATA_PATH = "dataset/train"     # 원본 데이터셋 루트
    OUTPUT_DATA_PATH = "trash_dataset_path"  # 분할된 데이터를 저장할 경로

    NUM_TRAIN = 100 # 0이면 전체 train 복사, >0이면 그 개수만큼 복사
    NUM_VAL_PERCENT = 1   # train 기준 %
    NUM_TEST_PERCENT = 1  # train 기준 %
    # ----------------------------------------

    # 폴더 생성
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUTPUT_DATA_PATH, split), exist_ok=True)

    # 클래스별 처리
    for class_name in os.listdir(ORIGINAL_DATA_PATH):
        class_path = os.path.join(ORIGINAL_DATA_PATH, class_name)
        if not os.path.isdir(class_path):
            continue

        # 샘플 무작위 섞기
        all_files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        random.shuffle(all_files)

        # ----------------- train 분할 -----------------
        if NUM_TRAIN == 0:
            train_files = all_files[:]
        else:
            train_files = all_files[:NUM_TRAIN] if len(all_files) >= NUM_TRAIN else all_files[:]

        # ----------------- val/test 분할 (train 기준, non-overlap) -----------------
        remaining_files = train_files[:]  # train 안에서 sampling

        val_count = math.floor(len(remaining_files) * NUM_VAL_PERCENT / 100)
        val_files = random.sample(remaining_files, val_count)
        remaining_files = [f for f in remaining_files if f not in val_files]

        test_count = math.floor(len(remaining_files) * NUM_TEST_PERCENT / 100)
        test_files = random.sample(remaining_files, test_count)

        # ----------------- 파일 복사 -----------------
        # train 복사
        dest_train_path = os.path.join(OUTPUT_DATA_PATH, "train", class_name)
        os.makedirs(dest_train_path, exist_ok=True)
        for file in train_files:
            shutil.copy2(os.path.join(class_path, file), dest_train_path)

        # val 복사
        dest_val_path = os.path.join(OUTPUT_DATA_PATH, "val", class_name)
        os.makedirs(dest_val_path, exist_ok=True)
        for file in val_files:
            shutil.copy2(os.path.join(class_path, file), dest_val_path)

        # test 복사
        dest_test_path = os.path.join(OUTPUT_DATA_PATH, "test", class_name)
        os.makedirs(dest_test_path, exist_ok=True)
        for file in test_files:
            shutil.copy2(os.path.join(class_path, file), dest_test_path)

    print(f"✅ 데이터셋 분할 완료. 결과는 '{OUTPUT_DATA_PATH}' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    print("🚨 원본 파일은 삭제하지 않습니다. 데이터를 백업해 두세요. 🚨")
    split_folder_fixed_count()
