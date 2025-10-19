# batch_resize.py
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm

sizes = [224, 299, 384]  # 원하는 크기 리스트
src_root = Path("dataset/orig")
dst_root = Path("dataset/resized")  # 결과 저장 경로

for s in sizes:
    dst_size_dir = dst_root / str(s)
    for class_dir in src_root.iterdir():
        if not class_dir.is_dir():
            continue
        out_class_dir = dst_size_dir / class_dir.name
        out_class_dir.mkdir(parents=True, exist_ok=True)
        for img_path in tqdm(list(class_dir.glob("*.*")), desc=f"Resizing {class_dir.name} -> {s}"):
            try:
                with Image.open(img_path) as im:
                    # 옵션: 세로/가로 비율 유지한 채로 resize 후 center crop
                    im = im.convert("RGB")
                    im = im.resize((s, s), resample=Image.BILINEAR)
                    out_path = out_class_dir / img_path.name
                    im.save(out_path, quality=95)
            except Exception as e:
                print("Error:", img_path, e)



###################################################### 
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def get_transforms(size, train=True):
    if train:
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ColorJitter(0.05,0.05,0.05,0.01),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
    else:
        return T.Compose([
            T.Resize(int(size*1.15)),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

# Usage
size = 224
train_ds = ImageFolder("dataset/resized/224/train", transform=get_transforms(size, train=True))
val_ds = ImageFolder("dataset/resized/224/val", transform=get_transforms(size, train=False))

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
