import os
import json
from tqdm import tqdm

# Sınıf isimleri ve sabit boyut
class_names = ["plane", "ship", "large-vehicle", "small-vehicle"]
WIDTH = 640
HEIGHT = 640

def find_image_file(images_dir, base_filename):
    for ext in [".png", ".jpg", ".jpeg"]:
        full_path = os.path.join(images_dir, base_filename + ext)
        if os.path.exists(full_path):
            return full_path
    return None  # Hiçbir görsel bulunamazsa None döner

def yolo_to_torchvision(label_dir, images_dir, output_json_path):
    dataset = []
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

    for idx, label_file in enumerate(tqdm(label_files, desc="Etiketler işleniyor")):
        base_name = os.path.splitext(label_file)[0]
        image_path = find_image_file(images_dir, base_name)

        if image_path is None:
            print(f"⚠️ Görsel bulunamadı: {base_name} (etiket var ama resim yok)")
            continue

        annotations = []
        label_path = os.path.join(label_dir, label_file)

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                class_id, x_center, y_center, box_w, box_h = map(float, parts)

                if box_w <= 0 or box_h <= 0:
                    continue  # ❗ Sıfır boyutlu kutuları atla

                x = (x_center - box_w / 2) * WIDTH
                y = (y_center - box_h / 2) * HEIGHT
                w = box_w * WIDTH
                h = box_h * HEIGHT

                annotations.append({
                    "bbox": [x, y, w, h],
                    "label": int(class_id)
                })

        dataset.append({
            "image_id": idx,
            "file_name": image_path,
            "width": WIDTH,
            "height": HEIGHT,
            "annotations": annotations
        })

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)

    print(f"✅ JSON başarıyla kaydedildi: {output_json_path}")


# 🔧 Örnek kullanım
if __name__ == "__main__":
    yolo_to_torchvision(
        label_dir=r"C:\Users\bilgi\Desktop\dota\train\labels",
        images_dir=r"C:\Users\bilgi\Desktop\dota\train\images",
        output_json_path=r"C:\Users\bilgi\Desktop\dota\train_annotations.json"
    )

    yolo_to_torchvision(
        label_dir=r"C:\Users\bilgi\Desktop\dota\val\labels",
        images_dir=r"C:\Users\bilgi\Desktop\dota\val\images",
        output_json_path=r"C:\Users\bilgi\Desktop\dota\val_annotations.json"
    )
