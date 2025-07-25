import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import wandb
import multiprocessing

KNN_DIR = "knn_outputs"
DATA_DIR = 'data'
IMG_SIZE_PIX = (64, 64)
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE_FEAT = 32

def load_pixel_data(split='train'):
    X = []
    y = []
    classes = sorted(os.listdir(os.path.join(DATA_DIR, split)))
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(DATA_DIR, split, cls)
        for fname in os.listdir(cls_dir):
            try:
                img = Image.open(os.path.join(cls_dir, fname)) \
                           .convert('RGB') \
                           .resize(IMG_SIZE_PIX)
                X.append(np.array(img).flatten())
                y.append(idx)
            except:
                continue
    return np.array(X), np.array(y), classes


# function to get the images and create for each of them better vectors representation using resnet18
def extract_features(split='train'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    ds = datasets.ImageFolder(os.path.join(DATA_DIR, split), transform=tf)
    loader = DataLoader(ds, batch_size=BATCH_SIZE_FEAT,shuffle=False, num_workers=1)
    cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    fe = nn.Sequential(*list(cnn.children())[:-1]).to(device).eval()
    feats, labs = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            out = fe(imgs).squeeze().cpu().numpy()
            feats.append(out)
            labs.append(labels.numpy())

    arr1=np.vstack(feats)
    arr2=np.vstack(labs)
    return arr1, arr2

def compute_and_log_metrics(y_true, y_pred, prefix):
    acc = np.mean(y_true == y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    wandb.log({
        f'{prefix}_accuracy': acc,
        f'{prefix}_precision': precision,
        f'{prefix}_recall': recall,
        f'{prefix}_f1': f1
    })
    return acc, precision, recall, f1

def print_top_bottom_classes(cm, classes, prefix):
    class_acc = cm.diagonal() / cm.sum(axis=1)
    sorted_idxs = np.argsort(class_acc)
    worst_idxs = sorted_idxs[:5]
    best_idxs = sorted_idxs[-5:][::-1]

    print(f"\n{prefix} — 5 Worst Classes:")
    for idx in worst_idxs:
        print(f"  {classes[idx]}: {class_acc[idx]:.4f}")
    print(f"\n{prefix} — 5 Best Classes:")
    for idx in best_idxs:
        print(f"  {classes[idx]}: {class_acc[idx]:.4f}")

def main():
    multiprocessing.freeze_support()
    os.makedirs(KNN_DIR, exist_ok=True)

    X, y, classes = load_pixel_data('train')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    Xf_train, yf_train = extract_features('train')
    Xf_test, yf_test = extract_features('valid')

    # hyperparams to try and find the best model
    grid = {
        'n_neighbors': [1, 3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    all_results = []
    for params in ParameterGrid(grid):
        run_name = f"knn_{params['n_neighbors']}_{params['weights']}_{params['metric']}"
        wandb.init(
            project="knn-classification",
            name=run_name,
            config=params,
            reinit=True
        )

        #  Raw-pixel try- just put straight up the images
        knn_raw = KNeighborsClassifier(**params)
        knn_raw.fit(X_train, y_train)
        y_pred_raw = knn_raw.predict(X_test)
        cm_raw = confusion_matrix(y_test, y_pred_raw)

        plt.figure(figsize=(6,6))
        plt.imshow(cm_raw, cmap='Blues')
        plt.title('Raw KNN Confusion Matrix')
        plt.axis('off')
        raw_cm_path = os.path.join(KNN_DIR, f"raw_cm_{run_name}.png")
        plt.savefig(raw_cm_path, dpi=200)
        plt.close()
        wandb.log({'raw_confusion_matrix': wandb.Image(raw_cm_path)})

        acc_raw, prec_raw, rec_raw, f1_raw = compute_and_log_metrics(y_test, y_pred_raw, 'raw')
        print(f"[{run_name} Raw] Acc={acc_raw},P={prec_raw}, R={rec_raw}, F1={f1_raw}")
        print_top_bottom_classes(cm_raw, classes, "Raw-pixel")

        # Feature-based try- move each image through resnet18 to get representation vector for each model
        knn_feat = KNeighborsClassifier(**params)
        knn_feat.fit(Xf_train, yf_train)
        y_pred_feat = knn_feat.predict(Xf_test)
        cm_feat = confusion_matrix(yf_test, y_pred_feat)

        plt.figure(figsize=(6,6))
        plt.imshow(cm_feat, cmap='Blues')
        plt.title('Feature KNN Confusion Matrix')
        plt.axis('off')
        feat_cm_path = os.path.join(KNN_DIR, f"feat_cm_{run_name}.png")
        plt.savefig(feat_cm_path, dpi=200)
        plt.close()
        wandb.log({'feat_confusion_matrix': wandb.Image(feat_cm_path)})

        acc_feat, prec_feat, rec_feat, f1_feat = compute_and_log_metrics(yf_test, y_pred_feat, 'feat')
        print(f"[{run_name} Feat] Acc={acc_feat},P={prec_feat}, R={rec_feat}, F1={f1_feat}")
        print_top_bottom_classes(cm_feat, classes, "Feature-based")

        all_results.append([
            run_name,
            acc_raw, prec_raw, rec_raw, f1_raw,
            acc_feat, prec_feat, rec_feat, f1_feat
        ])

        wandb.finish()

    summary_run = f"summary_{int(time.time())}"
    wandb.init(project="knn-classification", name=summary_run, reinit=True)
    table = wandb.Table(
        columns=[
            "run",
            "raw_accuracy", "raw_precision", "raw_recall", "raw_f1",
            "feat_accuracy", "feat_precision", "feat_recall", "feat_f1"
        ],
        data=all_results
    )
    wandb.log({"accuracy_precision_recall_f1_summary": table})
    wandb.finish()

    print("Overall Summary of All Runs:")
    for row in all_results:
        run = row[0]
        raw_metrics = row[1:5]
        feat_metrics = row[5:9]
        print(
            f"{run} | "
            f"Raw(Acc={raw_metrics[0]:.4f}, P={raw_metrics[1]:.4f}, "
            f"R={raw_metrics[2]:.4f}, F1={raw_metrics[3]:.4f}) | "
            f"Feat(Acc={feat_metrics[0]:.4f}, P={feat_metrics[1]:.4f}, "
            f"R={feat_metrics[2]:.4f}, F1={feat_metrics[3]:.4f})"
        )

if __name__ == '__main__':
    main()
