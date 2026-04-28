"""
generar_embeddings.py
=====================
Ejecuta este script UNA VEZ en local (o en Colab) antes de subir el repo a GitHub.
Genera embeddings.pkl a partir de los espectrogramas y lo guarda en la carpeta raíz.

Uso en Colab (pega en una celda):
    exec(open('generar_embeddings.py').read())
"""

import os, re, pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import torchvision.models as models

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────
MODELO_PATH        = "/content/drive/MyDrive/modelo_eff.pth"
ESPECTROGRAMAS_DIR = "/content/drive/MyDrive/dataset_Espectrogramas"
SALIDA             = "embeddings.pkl"   # se guarda en la carpeta actual
DIM                = 128
# ─────────────────────────────────────────────────────────────────────

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(2, 1, 1)[:8]),
    transforms.Normalize(mean=[0.5]*8, std=[0.5]*8),
])

class EfficientNetFingerprint(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        c = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            8, c.out_channels, c.kernel_size, c.stride, c.padding, bias=False
        )
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(self.backbone.classifier[1].in_features, dim),
        )
    def forward(self, x):
        return F.normalize(self.backbone(x), p=2, dim=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")

modelo = EfficientNetFingerprint(dim=DIM)
modelo.load_state_dict(torch.load(MODELO_PATH, map_location=device))
modelo = modelo.to(device).eval()
print("✅ Modelo cargado")

archivos = [f for f in os.listdir(ESPECTROGRAMAS_DIR) if f.endswith(".png")]
print(f"Procesando {len(archivos)} espectrogramas...")

grupos = {}
with torch.no_grad():
    for fname in tqdm(archivos):
        try:
            img = Image.open(os.path.join(ESPECTROGRAMAS_DIR, fname)).convert("RGBA")
            emb = modelo(TRANSFORM(img).unsqueeze(0).to(device)).cpu()
            nombre = fname.replace(".png", "")
            if " - " in nombre:
                nombre = nombre.split(" - ", 1)[1]
            nombre = re.sub(r"[\s_]*\(?V[-A-Z]+\)?$", "", nombre).strip()
            grupos.setdefault(nombre, []).append(emb)
        except Exception as e:
            print(f"  ⚠ {fname}: {e}")

db = {n: F.normalize(torch.stack(v).mean(0), p=2, dim=1) for n, v in grupos.items()}
with open(SALIDA, "wb") as f:
    pickle.dump(db, f)

print(f"\n✅ embeddings.pkl guardado con {len(db)} canciones → '{SALIDA}'")
print("Ya puedes subir este archivo al repositorio de GitHub.")
