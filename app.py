import io, os, re, pickle, tempfile, logging
from pathlib import Path
from contextlib import asynccontextmanager

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cover_detector")

# ── Rutas (relativas al repo) ─────────────────────────────────────────
BASE        = Path(__file__).parent
MODELO_PATH = BASE / "modelo.pth"
EMB_PATH    = BASE / "embeddings.pkl"
DIM         = 128
TOP_N       = 5

# ── Transformación (idéntica al entrenamiento) ────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(2, 1, 1)[:8]),
    transforms.Normalize(mean=[0.5]*8, std=[0.5]*8),
])


# ── Modelo ────────────────────────────────────────────────────────────
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


# ── Estado global ─────────────────────────────────────────────────────
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Dispositivo: {device}")

    log.info("Cargando modelo...")
    modelo = EfficientNetFingerprint(dim=DIM)
    modelo.load_state_dict(torch.load(MODELO_PATH, map_location=device))
    modelo.eval()
    state["modelo"] = modelo.to(device)
    state["device"] = device

    log.info("Cargando embeddings...")
    with open(EMB_PATH, "rb") as f:
        state["db"] = pickle.load(f)
    log.info(f"Listo — {len(state['db'])} canciones.")
    yield
    state.clear()


app = FastAPI(title="Cover Detector", lifespan=lifespan)


# ── Utilidades ────────────────────────────────────────────────────────
def audio_a_espectrograma(path: str) -> Image.Image:
    y, sr = librosa.load(path, sr=None)
    S = librosa.power_to_db(np.abs(librosa.stft(y)) ** 2, ref=np.max)
    buf = io.BytesIO()
    fig, ax = plt.subplots(figsize=(4, 4))
    librosa.display.specshow(S, sr=sr, x_axis=None, y_axis="log", ax=ax)
    ax.axis("off")
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")


# ── Endpoints ─────────────────────────────────────────────────────────
@app.post("/analizar")
async def analizar(audio: UploadFile = File(...)):
    ext = Path(audio.filename).suffix.lower()
    if ext not in {".mp3", ".wav", ".ogg", ".m4a", ".flac", ".webm"}:
        raise HTTPException(400, f"Formato no soportado: {ext}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await audio.read())
        ruta = tmp.name
    try:
        img = audio_a_espectrograma(ruta)
        tensor = TRANSFORM(img).unsqueeze(0).to(state["device"])
        with torch.no_grad():
            emb = state["modelo"](tensor)

        resultados = sorted(
            [{"cancion": n,
              "similitud": round(F.cosine_similarity(emb, e.to(state["device"])).item() * 100, 1)}
             for n, e in state["db"].items()],
            key=lambda x: x["similitud"], reverse=True
        )[:TOP_N]

        return JSONResponse({"resultados": resultados, "archivo": audio.filename})
    except Exception as e:
        log.error(f"Error: {e}")
        raise HTTPException(500, str(e))
    finally:
        os.unlink(ruta)


@app.get("/canciones")
async def canciones():
    return {"total": len(state["db"]), "canciones": sorted(state["db"].keys())}


@app.get("/", response_class=HTMLResponse)
async def index():
    return (BASE / "static" / "index.html").read_text(encoding="utf-8")


app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")
