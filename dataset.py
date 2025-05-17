# dataset.py

import torch
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from PIL import Image
import json

class CustomDataset(Dataset):
    def __init__(self, annotation_file, transforms=None):
        with open(annotation_file) as f:
            self.data = json.load(f)
        self.transforms = transforms

    def __getitem__(self, idx):
        sample = self.data[idx]
        img = Image.open(sample['file_name']).convert("RGB")
        img = ToTensor()(img)  # ✅ Tensor'a çeviriyoruz (PIL değil artık)

        boxes = []
        labels = []

        for ann in sample['annotations']:
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue  # ✅ Geçersiz (sıfır boyutlu) kutuları atla
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['label'] + 1)  # ✅ +1: background = 0 kabul edilir

        # ✅ Kutular boşsa, boş tensor döndür (PyTorch uyumlu)
        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([sample["image_id"]])
        }

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.data)
