#!/usr/bin/env python3
"""
batch_resize.py
의류 이미지 데이터셋을 224x224, 384x384 두 가지 해상도로
선명도를 최대한 유지하여 리사이즈하는 스크립트.
리사이즈 완료 후, 원본 파일을 삭제합니다.

사용법:
    python batch_resize.py --src dataset/orig --dst dataset/resized
"""

import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def resize_and_save(img_path: Path, out_path: Path, size: int) -> bool:
    """
    이미지 파일을 열고 정사각형(size x size)으로 리사이즈 후 저장.
    - LANCZOS 보간법 사용 → 디테일 보존
    - subsampling=0 → 색상 손실 최소화
    - quality=95 → 고화질 JPEG
    
    성공 시 True, 실패 시 False 반환.
    """
    try:
        with Image.open(img_path) as im:
            # 원본 이미지의 크기를 확인하여 리사이즈가 필요한지 판단 (생략 가능하나 안전성을 위해)
            # if im.size == (size, size):
            #     print(f"[SKIP] {img_path}는 이미 {size}x{size} 크기입니다.")
            #     return True

            im = im.convert("RGB")
            im = im.resize((size, size), resample=Image.LANCZOS)  # 고품질 리사이즈
            out_path.parent.mkdir(parents=True, exist_ok=True)
            im.save(out_path, quality=95, subsampling=0)  # 고화질 저장
            return True
    except Exception as e:
        print(f"[ERROR] {img_path} 처리 실패 → {e}")
        return False


def process_dataset(src_root: Path, dst_root: Path, sizes: list[int]):
    """
    src_root 안의 모든 이미지를 각 사이즈별로 리사이즈하여 저장합니다.
    모든 사이즈 리사이즈가 성공한 이미지에 대해서만 원본 파일을 삭제합니다.
    """
    # 원본 파일 목록을 가져옵니다.
    img_files = list(src_root.glob("*.*"))
    if not img_files:
        print(f"[WARN] {src_root} 안에 이미지 파일이 없습니다.")
        return

    # 모든 이미지 파일에 대해, 각 사이즈별 리사이즈 성공 여부를 저장할 딕셔너리
    # {Path('image.jpg'): {224: False, 384: False}} 형태
    success_status = {img_path: {size: False for size in sizes} for img_path in img_files}

    for size in sizes:
        print(f"\n[INFO] {size}x{size} 리사이즈 시작")
        dst_size_dir = dst_root / str(size)
        dst_size_dir.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(img_files, desc=f"Resizing {size}"):
            out_path = dst_size_dir / img_path.name
            
            # 리사이즈 및 저장 시도
            is_successful = resize_and_save(img_path, out_path, size)
            
            # 성공 여부 기록
            success_status[img_path][size] = is_successful

    
    # ----------------------------------------------------
    ## 원본 파일 삭제 로직
    print("\n[INFO] 모든 사이즈 리사이즈 완료 이미지 원본 삭제 시작...")
    
    deleted_count = 0
    
    for img_path, status in tqdm(success_status.items(), desc="Deleting Originals"):
        # 모든 사이즈에 대해 성공했는지 확인
        all_sizes_successful = all(status.values())
        
        if all_sizes_successful:
            try:
                # 원본 파일 삭제
                img_path.unlink()
                deleted_count += 1
            except Exception as e:
                print(f"[ERROR] 원본 파일 삭제 실패: {img_path} → {e}")
        else:
            # 하나라도 실패했다면, 해당 원본 파일은 유지합니다.
            # print(f"[SKIP] 원본 유지: {img_path} (모든 사이즈 리사이즈에 성공하지 못함)")
            pass

    print(f"\n✅ 원본 파일 {deleted_count}개 삭제 완료.")
    # ----------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="이미지 데이터셋 일괄 리사이즈 및 원본 삭제")
    parser.add_argument("--src", type=str, required=True, help="원본 데이터셋 폴더 경로")
    parser.add_argument("--dst", type=str, required=True, help="리사이즈된 데이터 저장 폴더")

    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    if not src_root.exists():
        raise FileNotFoundError(f"원본 폴더가 없습니다: {src_root}")

    # 224, 384 두 가지 사이즈만 처리
    # process_dataset(src_root, dst_root, sizes=[224, 384]) # 384 사이즈를 주석 해제하여 원래 목표대로 처리
    process_dataset(src_root, dst_root, sizes=[224])
    print("\n✅ 모든 리사이즈 및 삭제 작업 완료!")


if __name__ == "__main__":
    # main 함수 실행 전에 사용자에게 원본 삭제에 대한 경고를 줍니다.
    print("🚨 경고: 이 스크립트는 리사이즈 성공 후 원본 파일을 삭제합니다. 데이터를 백업했는지 확인하세요. 🚨")
    
    # 여기서 잠시 대기하거나 사용자 확인을 받을 수 있지만, 
    # 스크립트의 실행 흐름을 위해 바로 main을 호출합니다.
    main()