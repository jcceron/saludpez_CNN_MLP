# scripts/deap_makeEvalFuntion.py
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim
from sklearn.metrics import accuracy_score

from scripts.deap_MLPMultimodal import MLPMultimodal

# 1) Detectar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"==> Usando dispositivo: {device}")

# =========================
# 3) FUNCIÓN DE EVALUACIÓN PARA GA
# =========================
def make_eval_function(train_ds, 
                       val_ds, 
                       n_classes,
                       sensor_cols="sensores",
                       lr_bounds=(1e-4,1e-2),
                       bs_bounds=(16,64),
                       hd_bounds=(32,128)):
    """
    Crea y retorna la función eval_individual() que:
     - construye DataLoaders con el batch size del individuo
     - instancia y entrena el modelo 5 épocas con lr e hidden_dim del individuo
     - devuelve la accuracy en validación como fitness
    """
    def eval_individual(individual):
        # clamp de hiperparámetros
        lr = float(torch.clamp(torch.tensor(individual[0]), *lr_bounds))
        bs = int(torch.clamp(torch.tensor(individual[1]), *bs_bounds))
        hd = int(torch.clamp(torch.tensor(individual[2]), *hd_bounds))

        # DataLoaders con pin_memory si CUDA
        tr_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                               pin_memory=(device.type=="cuda"))
        vl_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                               pin_memory=(device.type=="cuda"))

        # Instanciar modelo y moverlo a device
        model = MLPMultimodal(
            img_size=(224,224),
            sensor_dim=len(sensor_cols),
            hidden_dim=hd,
            n_classes=n_classes
        ).to(device)

        opt = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        # Entrenar 5 épocas
        model.train()
        for _ in range(5):
            for imgs, sens, ys in tr_loader:
                imgs = imgs.to(device, non_blocking=True)
                sens = sens.to(device, non_blocking=True)
                ys   = ys.to(device, non_blocking=True)

                opt.zero_grad()
                out = model(imgs, sens)
                loss_fn(out, ys).backward()
                opt.step()

        # Evaluar
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for imgs, sens, ys in vl_loader:
                imgs = imgs.to(device, non_blocking=True)
                sens = sens.to(device, non_blocking=True)
                out = model(imgs, sens).argmax(dim=1).cpu().numpy()
                preds.extend(out); trues.extend(ys.numpy())

        return (accuracy_score(trues, preds),)
    return eval_individual