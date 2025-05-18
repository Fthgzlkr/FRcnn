import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torch.optim.lr_scheduler import StepLR
from dataset import CustomDataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
import os

def get_resnext_frcnn_model(num_classes=5):
    backbone = resnet_fpn_backbone("resnext50_32x4d", pretrained=True)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes)
    return model

def train_one_epoch(model, optimizer, loader, device):
    model.train()
    total_loss = 0.0
    for images, targets in tqdm(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"\n🟦 Ortalama Loss: {avg_loss:.4f}")

def evaluate_map(model, loader, device):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")

    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            ground_truths = [
                {"boxes": t["boxes"].cpu(), "labels": t["labels"].cpu()} for t in targets
            ]
            predictions = [
                {"boxes": o["boxes"].cpu(), "scores": o["scores"].cpu(), "labels": o["labels"].cpu()} for o in outputs
            ]

            metric.update(predictions, ground_truths)

    result = metric.compute()
    print("\n📊 Değerlendirme:")
    print(f"🔸 mAP@50: {result['map_50']:.4f} | mAP@75: {result['map_75']:.4f} | mAP: {result['map']:.4f} | mAR: {result['mar_100']:.4f}")
    return result['map_50']

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = CustomDataset("train_annotations.json")
    val_dataset = CustomDataset("val_annotations.json")

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    model = get_resnext_frcnn_model(num_classes=5).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.3)

    num_epochs = 50
    best_map = 0.0

    for epoch in range(1, num_epochs + 1):
        print(f"\n🚀 Epoch {epoch}/{num_epochs} başlıyor...")
        train_one_epoch(model, optimizer, train_loader, device)
        scheduler.step()

        if epoch % 5 == 0:
            current_map = evaluate_map(model, val_loader, device)
            if current_map > best_map:
                best_map = current_map
                torch.save(model.state_dict(), "best_resnext_frcnn.pth")
                print(f"🏆 En iyi model güncellendi! mAP@50: {best_map:.4f}")

        torch.save(model.state_dict(), f"resnext_frcnn_epoch_{epoch}.pth")
        print(f"💾 Model kaydedildi: resnext_frcnn_epoch_{epoch}.pth")

    print("\n✅ Eğitim tamamlandı!")
    os.system("shutdown /s /t 60")

if __name__ == "__main__":
    main()
