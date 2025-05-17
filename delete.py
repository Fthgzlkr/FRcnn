import os

def delete_npy_files(root_dir):
    deleted_files = 0
    for subdir in ['train/images', 'val/images']:
        dir_path = os.path.join(root_dir, subdir)
        if not os.path.exists(dir_path):
            print(f"🚫 Klasör bulunamadı: {dir_path}")
            continue

        for file in os.listdir(dir_path):
            if file.endswith(".npy"):
                file_path = os.path.join(dir_path, file)
                os.remove(file_path)
                deleted_files += 1
                print(f"🗑️ Silindi: {file_path}")

    print(f"\n✅ Toplam {deleted_files} .npy dosyası silindi.")

# 🔧 Kullanım:
# Aşağıdaki yolu kendi dataset dizinine göre güncelle:
dataset_root = r"C:\Users\bilgi\Desktop\dota"  # örnek yol
delete_npy_files(dataset_root)
