# scripts/deap_trainAndSaveBest.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
from sklearn.metrics import accuracy_score

from scripts.deap_MLPMultimodal import MLPMultimodal
# =========================
# 5) RE-ENTRENAR Y SERIALIZAR MEJOR MODELO
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_and_save_best(best, train_ds, val_ds, 
                        sensor_cols="sensores",
                        label_csv="labels2_kmeans_limpio.csv",
                        label_col="clases", 
                        modelo_path="modelos/best_model_deap.pt"):
    """
    - Reentrena 20 épocas mostrando barra de progreso
    - Imprime train_loss y val_acc en cada época
    - Guarda checkpoint cuando la val_acc mejore
    """
    lr, bs, hd = best
    bs, hd = int(bs), int(hd)

    tr_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                           pin_memory=(device.type=="cuda"))
    vl_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                           pin_memory=(device.type=="cuda"))

    model = MLPMultimodal(
        img_size=(224,224),
        sensor_dim=len(sensor_cols),
        hidden_dim=hd,
        n_classes=len(np.unique(pd.read_csv(label_csv)[label_col]))
    ).to(device)

    opt     = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    sched   = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', patience=3)

    best_acc = 0.0

    for epoch in trange(20, desc="Reentrenamiento"):
        # --- Entrenamiento ---
        model.train()
        running_loss = 0.0
        for imgs, sens, ys in tqdm(tr_loader, leave=False, desc=" Train batches"):
            imgs = imgs.to(device, non_blocking=True)
            sens = sens.to(device, non_blocking=True)
            ys   = ys.to(device, non_blocking=True)

            opt.zero_grad()
            out  = model(imgs, sens)
            loss = loss_fn(out, ys)
            loss.backward()
            opt.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(tr_loader)

        # --- Validación ---
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for imgs, sens, ys in tqdm(vl_loader, leave=False, desc=" Val batches"):
                imgs = imgs.to(device, non_blocking=True)
                sens = sens.to(device, non_blocking=True)
                out  = model(imgs, sens).argmax(dim=1).cpu().numpy()
                preds.extend(out)
                trues.extend(ys.numpy())

        val_acc = accuracy_score(trues, preds)

        # --- Scheduler y guardado de mejor ---
        sched.step(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), modelo_path)
            flag = "✅"
        else:
            flag = ""

        print(f"Epoch {epoch+1:02d} — train_loss: {avg_loss:.4f}, val_acc: {val_acc:.4f} {flag}")

    print(f"\n🏁 Reentrenamiento completo. Mejor val_acc: {best_acc:.4f}")