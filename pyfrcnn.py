# %%
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
from eval import evaluate

# %%

class CocoTransform:
    def __call__(self, image, target):
        image = F.to_tensor(image)  
        return image, target

# %%

def get_coco_dataset(img_dir, ann_file):
    return CocoDetection(
        root=img_dir,
        annFile=ann_file,
        transforms=CocoTransform()
    )


train_dataset = get_coco_dataset(
    img_dir="xray_knifes-6/train",
    ann_file="xray_knifes-6/train/_annotations.coco.json"
)


val_dataset = get_coco_dataset(
    img_dir="xray_knifes-6/valid",
    ann_file="xray_knifes-6/valid/_annotations.coco.json"
)




train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# %%

def get_model(num_classes):

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# %%

num_classes = 9 
model = get_model(num_classes)

# %%

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# %%
def train_one_epoch(model, optimizer, data_loader, device, epoch, index = 0):
    model.train()
    for images, targets in data_loader:
    
        images = [img.to(device) for img in images]

    
        processed_targets = []
        valid_images = []
        print(f"{index}")
        index += 1
        for i, target in enumerate(targets):
            boxes = []
            labels = []
        
            
            for obj in target:
             
          
                bbox = obj["bbox"]
                x, y, w, h = bbox

     
                if w > 0 and h > 0:
                    boxes.append([x, y, x + w, y + h]) 
                    labels.append(obj["category_id"])
            

  
            if boxes:
                processed_target = {
                    "boxes": torch.tensor(boxes, dtype=torch.float32).to(device),
                    "labels": torch.tensor(labels, dtype=torch.int64).to(device),
                }
                processed_targets.append(processed_target)
                valid_images.append(images[i])
 
        if not processed_targets:
            continue

      
        images = valid_images

        loss_dict = model(images, processed_targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"Epoch [{epoch}] Loss: {losses.item():.4f}")
    

# %%

num_epochs = 20
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, train_loader, device, epoch)
    lr_scheduler.step()
    epoch_mod = (epoch + 1) % 5
    if epoch_mod ==  0:
        metrics = evaluate(model, val_loader, device, num_classes)
        print("\nEvaluation Results:")
        print(f"mAP@50: {metrics['mAP_50']:.4f}")
        print(f"mAP@75: {metrics['mAP_75']:.4f}")
    model_path = f"fasterrcnn_resnet50_epoch_{epoch + 1}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")


