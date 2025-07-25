import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    top_k_accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import multiprocessing
import wandb

exp_name = "resnet101-pretrained"

def print_top_bottom_classes(cm, class_names, prefix):
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    sorted_idx = np.argsort(per_class_acc)
    worst = sorted_idx[:5]
    best  = sorted_idx[-5:][::-1]

    print(f"\n[{prefix}] 5 Worst Classes:")
    for i in worst:
        print(f"  {class_names[i]}: {per_class_acc[i]:.4f}")
    print(f"\n[{prefix}] 5 Best Classes:")
    for i in best:
        print(f"  {class_names[i]}: {per_class_acc[i]:.4f}")

def main():
    multiprocessing.freeze_support()
    wandb.init(
        project="cnn-model-comparison",
        name=exp_name,
        config={
            "batch_size": 32,
            "img_size": 224,
            "epochs": 30,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "model": "resnet50",
            "num_classes": 100
        }
    )
    config = wandb.config

    CNN_DIR = "cnn_outputs"
    os.makedirs(CNN_DIR, exist_ok=True)

    data_dir    = "data"
    batch_size  = config.batch_size
    img_size    = config.img_size
    num_classes = config.num_classes
    epochs      = config.epochs
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loaders
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "valid"), transform=val_tf)
    test_ds  = datasets.ImageFolder(os.path.join(data_dir, "test"),  transform=val_tf)
    class_names = train_ds.classes

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"=== Training {config.model} ===")
    model = models.resnet101(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model = model.to(DEVICE)

    wandb.watch(model, log="all", log_freq=100)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_val_acc = 0.0
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        train_correct = train_total = 0
        for imgs, labs in tqdm(train_loader, desc=f"Train {epoch}/{epochs}"):
            imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labs)
            loss.backward()
            optimizer.step()

            preds = out.argmax(1)
            train_correct += (preds == labs).sum().item()
            train_total += labs.size(0)
            train_loss += loss.item() * labs.size(0)

        train_acc = 100 * train_correct / train_total
        train_loss /= train_total

        model.eval()
        val_loss = 0.0
        val_correct = val_total = 0
        with torch.no_grad():
            for imgs, labs in val_loader:
                imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
                out = model(imgs)
                loss = criterion(out, labs)

                preds = out.argmax(1)
                val_correct += (preds == labs).sum().item()
                val_total += labs.size(0)
                val_loss += loss.item() * labs.size(0)

        val_acc = 100 * val_correct / val_total
        val_loss /= val_total

        print(f"Epoch {epoch}:Train Acc={train_acc}% Loss={train_loss} | Val Acc={val_acc}% Loss={val_loss}")
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(CNN_DIR, f"best_{exp_name}.pth")
            torch.save(model.state_dict(), ckpt_path)
            artifact = wandb.Artifact(f"{config.model}-model", type="model")
            artifact.add_file(ckpt_path)
            wandb.log_artifact(artifact)

    wandb.run.summary["best_val_acc"] = best_val_acc
    print(f"Best accuracy: {best_val_acc}%")

    model.load_state_dict(torch.load(os.path.join(CNN_DIR, f"best_{exp_name}.pth")))
    model.eval()
    all_preds =[]
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for imgs, labs in test_loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labs.numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    test_top1 = (all_preds == all_labels).mean() * 100
    test_top5 = top_k_accuracy_score(all_labels, all_probs, k=5) * 100

    test_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    test_rec  = recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    test_f1   = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100

    wandb.log({
        "test_top1":   test_top1,
        "test_top5":   test_top5,
        "test_precision": test_prec,
        "test_recall":    test_rec,
        "test_f1":        test_f1
    })

    print(f"\nTest Results: Top-1={test_top1}%, Top-5={test_top5}%, P={test_prec}%, R={test_rec}%, F1={test_f1}%")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,8))
    plt.imshow(cm, cmap='Blues')
    plt.title(f'{config.model} Confusion Matrix')
    plt.colorbar()
    plt.tight_layout()
    cm_path = os.path.join(CNN_DIR, f"{exp_name}_cm.png")
    plt.savefig(cm_path, dpi=200)
    wandb.log({"confusion_matrix": wandb.Image(cm_path)})

    print_top_bottom_classes(cm, class_names, exp_name)

    # plt.figure(figsize=(10,5))
    # plt.plot([1, epochs], [test_top1, test_top1], 'o-', label='Top-1 Acc')
    # plt.plot([1, epochs], [test_top5, test_top5], 's--', label='Top-5 Acc')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy (%)')
    # plt.title(f'{exp_name} Performance')
    # plt.legend()
    # summary_path = os.path.join(CNN_DIR, f'model_performance_{exp_name}.png')
    # plt.savefig(summary_path, dpi=200)
    # wandb.log({"performance_summary": wandb.Image(summary_path)})

if __name__ == "__main__":
    main()
