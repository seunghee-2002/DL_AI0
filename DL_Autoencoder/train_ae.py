import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    explained_variance_score
)
from tqdm import tqdm
from paths import *

# — hyperparams —
BATCH_SIZE = 4096
LR         = 1e-3
WD         = 1e-5
EPOCHS     = 100
VAL_RATIO  = 0.2
PATIENCE   = 5   # early stopping patience

# 1) 데이터 & split
X = np.load(OUT_DIR + "ev_concat176.npy").astype(np.float32)
dataset = TensorDataset(torch.tensor(X))
n_val    = int(len(dataset) * VAL_RATIO)
n_train  = len(dataset) - n_val
train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                generator=torch.Generator().manual_seed(0))
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

# 2) AE 모델 정의
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(176, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            
            nn.Linear(64, 16),   # bottleneck
        )
        self.dec = nn.Sequential(
            nn.Linear(16, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            
            nn.Linear(128, 176),
        )

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z


device = "cuda" if torch.cuda.is_available() else "cpu"
model  = AE().to(device)
opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.2)
crit   = nn.MSELoss()

best_val_mse = float('inf')
best_epoch   = 0
no_improve   = 0

for epoch in range(1, EPOCHS+1):
    # — Training —
    model.train()
    train_loss = 0.0
    for (batch,) in train_dl:
        batch = batch.to(device)
        opt.zero_grad()
        out, _ = model(batch)
        loss = crit(out, batch)
        loss.backward()
        opt.step()
        train_loss += loss.item() * batch.size(0)
    train_loss /= n_train

    # — Validation —
    model.eval()
    val_preds, val_truths = [], []
    with torch.no_grad():
        for (batch,) in val_dl:
            batch = batch.to(device)
            out, _ = model(batch)
            val_preds.append(out.cpu().numpy())
            val_truths.append(batch.cpu().numpy())
    val_preds  = np.vstack(val_preds)
    val_truths = np.vstack(val_truths)

    # Metrics
    val_mse = mean_squared_error(val_truths, val_preds)
    val_mae = mean_absolute_error(val_truths, val_preds)
    val_r2  = r2_score(val_truths, val_preds)
    val_ev  = explained_variance_score(val_truths, val_preds)

    print(f"[Epoch {epoch:03d}] "
          f"Train MSE: {train_loss:.4f} | "
          f"Val MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, "
          f"R²: {val_r2:.4f}, EV: {val_ev:.4f}")

    # — Early Stopping & Checkpoint —
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_epoch   = epoch
        no_improve   = 0
        torch.save(model.state_dict(), OUT_DIR + "ae_final16_best.pt")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"⏹ Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

print(f"🏆 Best Val MSE {best_val_mse:.4f} at epoch {best_epoch}")
