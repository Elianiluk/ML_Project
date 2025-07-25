import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# Configuration
DATA_DIR = 'data'  # expects train/, valid/, test/ subdirs
IMG_SIZE = (128, 128)
SAMPLES_PER_CLASS = 5

# 1. Compute class distribution
def compute_class_distribution(split='train'):
    classes = sorted(os.listdir(os.path.join(DATA_DIR, split)))
    counts = {}
    for cls in classes:
        cls_dir = os.path.join(DATA_DIR, split, cls)
        counts[cls] = len(os.listdir(cls_dir))
    return counts

# 2. Compute image size stats
def compute_image_stats(split='train'):
    widths, heights = [], []
    for root, _, files in os.walk(os.path.join(DATA_DIR, split)):
        for fname in files:
            try:
                img = Image.open(os.path.join(root, fname))
                w, h = img.size
                widths.append(w)
                heights.append(h)
            except:
                continue
    return (np.mean(widths), np.std(widths)), (np.mean(heights), np.std(heights))

# 3. Display sample images per class
def show_samples(split='train'):
    classes = sorted(os.listdir(os.path.join(DATA_DIR, split)))
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor()
    ])

    plt.figure(figsize=(15, len(classes)*2))
    for i, cls in enumerate(classes):
        cls_dir = os.path.join(DATA_DIR, split, cls)
        files = os.listdir(cls_dir)
        samples = random.sample(files, min(SAMPLES_PER_CLASS, len(files)))
        for j, fname in enumerate(samples):
            img = Image.open(os.path.join(cls_dir, fname)).convert('RGB')
            img = img.resize(IMG_SIZE)
            plt_idx = i * SAMPLES_PER_CLASS + j + 1
            ax = plt.subplot(len(classes), SAMPLES_PER_CLASS, plt_idx)
            ax.imshow(img)
            ax.axis('off')
            if j == 0:
                ax.set_title(cls)
    plt.tight_layout()
    plt.savefig('dataset_samples.png', dpi=200)
    print('Saved dataset_samples.png')

# 4. Create summary DataFrame
def create_summary_df(split='train'):
    counts = compute_class_distribution(split)
    (w_mean, w_std), (h_mean, h_std) = compute_image_stats(split)
    df = pd.DataFrame(list(counts.items()), columns=['class', 'count'])
    df['mean_width'] = w_mean
    df['std_width'] = w_std
    df['mean_height'] = h_mean
    df['std_height'] = h_std
    return df

if __name__ == '__main__':
    for split in ['train', 'valid', 'test']:
        print(f"\n=== {split.upper()} split distribution ===")
        dist = compute_class_distribution(split)
        for cls, cnt in dist.items(): print(f"{cls}: {cnt}")
        (w_m, w_s), (h_m, h_s) = compute_image_stats(split)
        print(f"Image width: mean={w_m:.1f}, std={w_s:.1f}")
        print(f"Image height: mean={h_m:.1f}, std={h_s:.1f}")

    # Show samples from train
    show_samples('train')
    # Save summary
    import pandas as pd
    df = create_summary_df('train')
    df.to_csv('dataset_summary_train.csv', index=False)
    print('Saved dataset_summary_train.csv')
