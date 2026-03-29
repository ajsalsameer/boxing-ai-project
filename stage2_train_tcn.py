"""
╔══════════════════════════════════════════════════════════════╗
║    BOXING AI - STAGE 2: TCN RECOGNITION TRAINER              ║
║    5 classes: jab, cross, hook, uppercut, idle               ║
╚══════════════════════════════════════════════════════════════╝

Run:   python stage2_train_tcn.py

⚠️  BEFORE RUNNING: collect idle samples with live_trainer.py
    (hold I while standing still — aim for 80+ samples)

Requirements:
    pip install torch numpy scikit-learn matplotlib seaborn
"""

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np, json, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from models_def import (
    TCN_Boxing, CLASSES, NUM_CLASSES, IDLE_IDX,
    LANDMARKS_USED, SEQ_LEN, INPUT_SIZE, BASE_FEATURES,
    compute_motion_features,
)

CONFIG_PATH = Path("dataset/config.json")
S1_CONFIG   = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {
    "window_size": SEQ_LEN, "classes": CLASSES,
    "landmarks_used": LANDMARKS_USED, "output_dir": "dataset"
}

TRAIN_CFG = {
    "batch_size": 32, "epochs": 100, "lr": 0.001,
    "val_split": 0.2, "model_dir": "models",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
print(f"🖥️  Device: {TRAIN_CFG['device']}")
if TRAIN_CFG["device"] == "cuda":
    print(f"    GPU: {torch.cuda.get_device_name(0)}")


# ─────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────
class BoxingDataset(Dataset):
    def __init__(self, data_dir="dataset"):
        self.samples, self.labels = [], []
        for idx, cls in enumerate(CLASSES):
            d = Path(data_dir) / cls
            if not d.exists():
                print(f"  ⚠️  {d} not found — skipping")
                continue
            files = list(d.glob("*.npy"))
            print(f"  📂 [{cls}]: {len(files)} files")
            skipped = 0
            for f in files:
                arr = np.load(str(f))
                if arr.shape == (SEQ_LEN, BASE_FEATURES):
                    motion = compute_motion_features(arr)
                    self.samples.append(motion.astype(np.float32))
                    self.labels.append(idx)
                else:
                    skipped += 1
            if skipped:
                print(f"     ⚠️  Skipped {skipped} files — wrong shape")
            loaded = sum(1 for l in self.labels if l == idx)
            print(f"     ✅ Loaded {loaded} samples")

        print(f"\n  Total: {len(self.samples)} samples across {NUM_CLASSES} classes\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return torch.tensor(self.samples[i]).T, torch.tensor(self.labels[i], dtype=torch.long)


# ─────────────────────────────────────────────
#  TRAIN
# ─────────────────────────────────────────────
def train():
    print("╔══════════════════════════════════════════╗")
    print("║   BOXING AI — Stage 2: Training TCN      ║")
    print(f"║   Classes ({NUM_CLASSES}): {', '.join(CLASSES)}")
    print("╚══════════════════════════════════════════╝\n")

    print("📦 Loading dataset...")
    ds = BoxingDataset(S1_CONFIG.get("output_dir", "dataset"))

    if len(ds) < 10:
        print("❌ Not enough samples! Collect data with live_trainer.py first.")
        return

    # Check idle exists — it's now mandatory
    label_arr    = np.array(ds.labels)
    class_counts = np.bincount(label_arr, minlength=NUM_CLASSES)

    if class_counts[IDLE_IDX] < 10:
        print(f"❌ Only {class_counts[IDLE_IDX]} idle samples found.")
        print("   Run live_trainer.py and hold 'I' while standing still (80+ samples needed).")
        return

    print("\n  Samples per class:")
    for i, cls in enumerate(CLASSES):
        target = 80 if cls == "idle" else 60
        bar    = "█" * (class_counts[i] // 10)
        status = "✅" if class_counts[i] >= target else ("⚠️ " if class_counts[i] >= target//2 else "❌ LOW")
        print(f"    {status} {cls:<10}: {class_counts[i]:4d}  {bar}")

    # Class weights — inverse frequency, keeps idle from dominating
    # idle typically has more samples so it would get lower weight automatically
    raw_w   = 1.0 / np.maximum(class_counts, 1).astype(float)
    weights = torch.tensor(raw_w / raw_w.sum() * NUM_CLASSES, dtype=torch.float32)
    print(f"\n  Class weights: {dict(zip(CLASSES, weights.numpy().round(3)))}\n")

    val_sz  = int(len(ds) * TRAIN_CFG["val_split"])
    trn_sz  = len(ds) - val_sz
    trn_ds, val_ds = random_split(ds, [trn_sz, val_sz])
    trn_dl  = DataLoader(trn_ds, batch_size=TRAIN_CFG["batch_size"], shuffle=True,  num_workers=0)
    val_dl  = DataLoader(val_ds, batch_size=TRAIN_CFG["batch_size"], shuffle=False, num_workers=0)
    print(f"  Train: {trn_sz} | Val: {val_sz}\n")

    device = TRAIN_CFG["device"]
    model  = TCN_Boxing(input_size=INPUT_SIZE, num_classes=NUM_CLASSES).to(device)
    print(f"🧠 TCN params: {sum(p.numel() for p in model.parameters()):,}  ({NUM_CLASSES} output classes)\n")

    # ── CrossEntropyLoss + label smoothing ───────────────────────
    # label_smoothing=0.1 softens cross/hook boundary without
    # hurting idle. Focal loss removed — backfires at low idle
    # sample counts by down-weighting idle when it needs help most.
    criterion = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.1)
    print(f"  Loss: CrossEntropyLoss (label_smoothing=0.1)\n")
    optimizer = optim.AdamW(model.parameters(), lr=TRAIN_CFG["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, TRAIN_CFG["epochs"])

    Path(TRAIN_CFG["model_dir"]).mkdir(exist_ok=True)
    best_val_acc = 0.0
    trn_losses, val_losses, trn_accs, val_accs = [], [], [], []

    print(f"🏋️  Training {TRAIN_CFG['epochs']} epochs...\n")
    for ep in range(1, TRAIN_CFG["epochs"] + 1):
        model.train()
        t_loss = t_cor = 0
        for x, y in trn_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(x)
            loss = criterion(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
            t_cor  += (out.argmax(1) == y).sum().item()
        scheduler.step()
        t_acc  = t_cor / trn_sz
        t_loss /= len(trn_dl)

        model.eval()
        v_loss = v_cor = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y  = x.to(device), y.to(device)
                out   = model(x)
                loss  = criterion(out, y)
                v_loss += loss.item()
                v_cor  += (out.argmax(1) == y).sum().item()
        v_acc  = v_cor / val_sz
        v_loss /= len(val_dl)

        trn_losses.append(t_loss); val_losses.append(v_loss)
        trn_accs.append(t_acc);   val_accs.append(v_acc)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save({
                "epoch": ep, "model_state": model.state_dict(),
                "val_acc":    v_acc,
                "classes":    CLASSES,
                "input_size": INPUT_SIZE,
                "seq_len":    SEQ_LEN,
                "num_classes": NUM_CLASSES,
                "config":     S1_CONFIG,
            }, f"{TRAIN_CFG['model_dir']}/tcn_boxing.pth")
            marker = " ← BEST"
        else:
            marker = ""

        if ep % 10 == 0 or ep == 1:
            print(f"  Ep {ep:3d} | Loss {t_loss:.4f}/{v_loss:.4f} | Acc {t_acc:.1%}/{v_acc:.1%}{marker}")

    print(f"\n✅ Best val acc: {best_val_acc:.1%}")
    print(f"💾 Saved: {TRAIN_CFG['model_dir']}/tcn_boxing.pth\n")

    # Per-class evaluation
    ckpt = torch.load(f"{TRAIN_CFG['model_dir']}/tcn_boxing.pth", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in val_dl:
            preds.extend(model(x.to(device)).argmax(1).cpu().numpy())
            labels.extend(y.numpy())

    print(classification_report(labels, preds, target_names=CLASSES))
    print("\n  💡 If cross/hook confusion persists:")
    print("     1. Record 20+ more cross samples from defensive stance")
    print("     2. Record 20+ more hook samples with exaggerated elbow position")
    print("     3. Retrain — idle class boundary will tighten up both of them\n")

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(trn_losses, label="Train"); axes[0].plot(val_losses, label="Val")
    axes[0].legend(); axes[0].set_title("Loss")
    axes[1].plot([a*100 for a in trn_accs], label="Train")
    axes[1].plot([a*100 for a in val_accs], label="Val")
    axes[1].set_title("Accuracy (%)"); axes[1].legend(); axes[1].set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(f"{TRAIN_CFG['model_dir']}/training_results.png", dpi=120)

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f"Confusion Matrix — {NUM_CLASSES} classes"); plt.tight_layout()
    plt.savefig(f"{TRAIN_CFG['model_dir']}/confusion_matrix.png", dpi=120)
    print(f"📈 Graphs saved to {TRAIN_CFG['model_dir']}/")


if __name__ == "__main__":
    train()