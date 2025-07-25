import os
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate
import xgboost as xgb
import wandb
import kagglehub
import itertools

# ====== Paths and config ======
DATA_DIR = kagglehub.dataset_download("gpiosenka/sports-classification")
CACHE_DIR = "cache_feats"
OUTPUT_DIR = "xgb_outputs"
BATCH_SIZE = 32
N_FOLDS = 5

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====== Hyperparameter grid (2^6 = 64 runs) ======
PARAM_GRID = {
    'learning_rate':    [0.01, 0.1],
    'n_estimators':     [50, 100],
    'max_depth':        [3, 6],
    'subsample':        [0.8, 1.0],
    'colsample_bytree': [0.6, 0.8],
    'min_child_weight': [1, 5]
}

# ====== Logger helper ======
_start = time.perf_counter()
def log(msg):
    print(f"[{time.perf_counter() - _start:6.1f}s] {msg}")

# ====== Feature extraction via ResNet18 ======
def get_features(split="train"):
    path = os.path.join(CACHE_DIR, f"{split}_resnet18.npz")
    if os.path.exists(path):
        data = np.load(path)
        log(f"Loaded cached {split} features {data['X'].shape}")
        return data["X"], data["y"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = (transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2,0.2,0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ]) if split=="train"
        else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    )

    ds = datasets.ImageFolder(os.path.join(DATA_DIR, split), transform=tf)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=2, pin_memory=True)

    cnn = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1
    ).to(device).eval()
    feat_extractor = nn.Sequential(*list(cnn.children())[:-1])

    feats, labs = [], []
    with torch.no_grad():
        for imgs, lbls in dl:
            imgs = imgs.to(device, non_blocking=True)
            out = feat_extractor(imgs).squeeze(-1).squeeze(-1).cpu().numpy()
            feats.append(out)
            labs.append(lbls.numpy())

    X = np.vstack(feats).astype(np.float32)
    y = np.hstack(labs).astype(np.int64)
    np.savez_compressed(path, X=X, y=y)
    log(f"Cached {split} features {X.shape}")
    return X, y

# ====== Evaluation function ======
def evaluate_xgb(X_train, y_train, X_test, y_test, params, run_name):
    # 1) Cross-validation (no early stopping)
    cv_clf = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss",
        tree_method="gpu_hist" if torch.cuda.is_available() else "hist",
        n_jobs=-1,
        **params
    )
    cv_results = cross_validate(
        cv_clf, X_train, y_train, cv=N_FOLDS,
        scoring=['accuracy','precision_macro','recall_macro','f1_macro'],
        n_jobs=-1
    )

    # 2) Full fit with early stopping
    clf = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss",
        tree_method="gpu_hist" if torch.cuda.is_available() else "hist",
        early_stopping_rounds=10,
        n_jobs=-1,
        **params
    )
    t0 = time.perf_counter()
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    log(f"{run_name} fit in {time.perf_counter() - t0:.1f}s")

    preds = clf.predict(X_test)
    metrics = {
        'accuracy':     accuracy_score(y_test, preds),
        'precision':    precision_score(y_test, preds, average='macro', zero_division=0),
        'recall':       recall_score(y_test, preds, average='macro', zero_division=0),
        'f1':           f1_score(y_test, preds, average='macro', zero_division=0),
        'cv_accuracy':  np.mean(cv_results['test_accuracy']),
        'cv_precision': np.mean(cv_results['test_precision_macro']),
        'cv_recall':    np.mean(cv_results['test_recall_macro']),
        'cv_f1':        np.mean(cv_results['test_f1_macro'])
    }

    wandb.log({
        f"{run_name}_accuracy":      metrics['accuracy'],
        f"{run_name}_precision":     metrics['precision'],
        f"{run_name}_recall":        metrics['recall'],
        f"{run_name}_f1":            metrics['f1'],
        f"{run_name}_cv_accuracy":   metrics['cv_accuracy'],
        f"{run_name}_cv_precision":  metrics['cv_precision'],
        f"{run_name}_cv_recall":     metrics['cv_recall'],
        f"{run_name}_cv_f1":         metrics['cv_f1']
    })
    return metrics

# ====== Main ======
def main():
    log("Starting XGBoost hyperparameter tuning")
    run = wandb.init(project="xgb-classification", name="xgb_hyperparam_tuning")

    try:
        X_train_feat, y_train = get_features("train")
        X_test_feat,  y_test  = get_features("valid")

        columns = ['run_name','accuracy','precision','recall','f1']
        table = wandb.Table(columns=columns)

        for (lr, n_est, max_d, subs, colbt, mcw) in itertools.product(
            PARAM_GRID['learning_rate'],
            PARAM_GRID['n_estimators'],
            PARAM_GRID['max_depth'],
            PARAM_GRID['subsample'],
            PARAM_GRID['colsample_bytree'],
            PARAM_GRID['min_child_weight']
        ):
            params = {
                'learning_rate':    lr,
                'n_estimators':     n_est,
                'max_depth':        max_d,
                'subsample':        subs,
                'colsample_bytree': colbt,
                'min_child_weight': mcw
            }
            run_name = f"xgb_lr{lr}_n{n_est}_d{max_d}_sub{subs}_col{colbt}_mcw{mcw}"
            log(f"Evaluating {run_name}")

            metrics = evaluate_xgb(
                X_train_feat, y_train,
                X_test_feat,  y_test,
                params, run_name
            )

            table.add_data(
                run_name,
                metrics['accuracy'],
                metrics['precision'],
                metrics['recall'],
                metrics['f1']
            )

        wandb.log({"xgb_metrics_table": table})
    finally:
        run.finish()
        log("Done.")

if __name__ == "__main__":
    main()
