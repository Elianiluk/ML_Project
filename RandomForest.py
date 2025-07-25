import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import wandb
import multiprocessing

OUT_DIR = "rf_outputs"
os.makedirs(OUT_DIR, exist_ok=True)
DATA_DIR = 'data'
IMG_SIZE = (64, 64)
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 32

# function to get the images and create for each of them better vectors representation using resnet18
def extract_features(split='train'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])
    ds = datasets.ImageFolder(os.path.join(DATA_DIR, split), transform=tf)

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
    cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    feat_ext = nn.Sequential(*list(cnn.children())[:-1]).to(device).eval()

    feats = []
    labs = []
    with torch.no_grad():
        for imgs, labels in loader:
            out = feat_ext(imgs.to(device)).squeeze().cpu().numpy()
            feats.append(out)
            labs.append(labels.numpy())

    arr1 = np.vstack(feats)
    arr2 = np.vstack(labs)
    return arr1, arr2

def print_top_bottom_classes(cm, classes, prefix):
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    sorted_idx = np.argsort(per_class_acc)
    worst = sorted_idx[:5]
    best  = sorted_idx[-5:][::-1]

    print(f"\n[{prefix}] 5 Worst Classes:")
    for i in worst:
        print(f"  {classes[i]}: {per_class_acc[i]:.4f}")
    print(f"\n[{prefix}] 5 Best Classes:")
    for i in best:
        print(f"  {classes[i]}: {per_class_acc[i]:.4f}")

def main():
    multiprocessing.freeze_support()

    X_train, y_train = extract_features('train')
    X_test,  y_test  = extract_features('valid')
    classes = sorted(os.listdir(os.path.join(DATA_DIR, 'train')))

    # hyperparams to try and find the best model
    grid = {
        'n_estimators':    [50, 100, 200],
        'max_depth':       [None, 10, 20],
        'min_samples_leaf':[1, 5, 10],
        'max_features':    ['sqrt', 'log2']
    }

    all_results = []
    for params in ParameterGrid(grid):
        run_name = (f"RF_e{params['n_estimators']}_d{params['max_depth']}_l{params['min_samples_leaf']}_f{params['max_features']}")
        wandb.init(
            project="rf_classification",
            name=run_name,
            config=params,
            reinit=True
        )

        rf = RandomForestClassifier(random_state=RANDOM_STATE,n_jobs=1,**params)
        rf.fit(X_train, y_train)

        preds = rf.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        precision  = precision_score(y_test, preds, average='macro', zero_division=0)
        recall   = recall_score(y_test, preds, average='macro', zero_division=0)
        f1    = f1_score(y_test, preds, average='macro', zero_division=0)

        wandb.log({
            'accuracy':  acc,
            'precision': precision,
            'recall':    recall,
            'f1_score':  f1
        })

        print(f"[{run_name}] Acc={acc}, P={precision}, R={recall}, F1={f1}")

        # confusion matrix
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(6,6))
        plt.imshow(cm, cmap='Blues')
        plt.axis('off')
        cm_path = os.path.join(OUT_DIR, f"cm_{run_name}.png")
        plt.savefig(cm_path, dpi=200)
        plt.close()
        wandb.log({'confusion_matrix': wandb.Image(cm_path)})

        print_top_bottom_classes(cm, classes, run_name)

        # get the 20 most important features
        imps = rf.feature_importances_
        top20 = np.argsort(imps)[-20:]
        plt.figure(figsize=(10,6))
        plt.barh(range(len(top20)), imps[top20])
        plt.yticks(range(len(top20)), [f"f{idx}" for idx in top20])
        plt.title("Top 20 Feature Importances")
        fi_path = os.path.join(OUT_DIR, f"fi_{run_name}.png")
        plt.savefig(fi_path, dpi=200)
        plt.close()
        wandb.log({'feature_importances': wandb.Image(fi_path)})

        all_results.append([run_name, acc, precision, recall, f1])
        wandb.finish()

    summary_name = f"rf_summary_{int(time.time())}"
    wandb.init(project="rf_classification", name=summary_name, reinit=True)
    table = wandb.Table(
        columns=['run','accuracy','precision','recall','f1'],
        data=all_results
    )
    wandb.log({'summary': table})
    wandb.finish()

    print("\nSummary of all runs:")
    for r, a, p, rcl, f in all_results:
        print(f"{r}: acc={a:.4f}, prec={p:.4f}, rec={rcl:.4f}, f1={f:.4f}")

if __name__ == "__main__":
    main()
