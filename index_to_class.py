import os
import pandas as pd

# 1) Point to your actual train folder:
train_dir = "data/train"

# 2) List & sort the 100 class‑folder names:
class_names = sorted(os.listdir(train_dir))

# 3) Build a DataFrame
df = pd.DataFrame({
    "index": range(len(class_names)),
    "class_name": class_names
})

# 4) Save out the complete CSV
out_path = "sports_classes_full.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(class_names)} classes to {out_path}")
