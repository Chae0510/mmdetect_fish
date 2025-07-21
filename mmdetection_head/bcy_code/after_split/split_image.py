import os
import shutil
import random

def split_and_copy_images(source_dir, dest_root, train_ratio=0.7, val_ratio=0.2, seed=42):
    random.seed(seed)
    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.png'))]
    random.shuffle(image_files)

    total = len(image_files)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)

    splits = {
        'train': image_files[:n_train],
        'val': image_files[n_train:n_train + n_val],
        'test': image_files[n_train + n_val:]
    }

    for split, files in splits.items():
        dest_dir = os.path.join(dest_root, split)
        os.makedirs(dest_dir, exist_ok=True)
        for f in files:
            shutil.copy(os.path.join(source_dir, f), os.path.join(dest_dir, f))

    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    return splits

if __name__ == "__main__":
    splits = split_and_copy_images(
        source_dir='/workspace/ori_images',
        dest_root='/workspace/fish_data',
        train_ratio=0.7,
        val_ratio=0.2
    )

    for split, files in splits.items():
        with open(f"{split}_files.txt", "w") as f:
            f.write("\n".join(files))
