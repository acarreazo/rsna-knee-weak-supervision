"""Configuración común del pipeline P1 (RSNA Knee Abnormality Detection).

Todas las rutas y constantes viven aquí. Los scripts numerados (01_..., 02_...) la importan.

Ejecución típica en un Kaggle Notebook (GPU T4, Internet ON solo si se usa el modelo NLI o pesos
preentrenados):  los datos de la competencia se montan en /kaggle/input/<slug>/ y las salidas
van a /kaggle/working/p1/.

En local (prueba con datos sintéticos) se fija la variable de entorno P1_DATA_ROOT.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------------------------
# Etiquetas: nombres EXACTOS de las columnas de train.csv (verificados en la pestaña Data y en
# los repositorios públicos de la competencia). Ver 01_decisiones/D002.
# ----------------------------------------------------------------------------------------------
LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
LABELS_ES = {
    "ACL": "LCA", "MCL": "Lig. colateral medial", "Medial Meniscus": "Menisco medial",
    "Lateral Meniscus": "Menisco lateral", "Medial OA": "Artrosis medial",
    "Lateral OA": "Artrosis lateral", "PF OA": "Artrosis patelofemoral",
    "Effusion": "Derrame", "Synovitis": "Sinovitis", "Baker's": "Quiste de Baker",
    "Contusion": "Contusión ósea", "Fracture": "Fractura",
}
N_LABELS = len(LABELS)

STUDY_COL = "StudyInstanceUID"
SERIES_COL = "SeriesInstanceUID"
REPORT_COL = "Report"
PLANE_COL = "Anatomical_Plane"      # Sagittal / Coronal / Axial
FLUID_COL = "Fluid_Sensitive"       # 1 = PD/T2 FS (sensible a líquido)

SEED = 42


# ----------------------------------------------------------------------------------------------
# Rutas
# ----------------------------------------------------------------------------------------------
def _find_data_root() -> Path:
    env = os.environ.get("P1_DATA_ROOT")
    if env:
        return Path(env)
    base = Path("/kaggle/input")
    if base.is_dir():
        # el slug puede montarse a 1, 2 o 3 niveles: buscamos la carpeta que tenga train.csv
        for depth in range(1, 4):
            for p in base.glob("/".join(["*"] * depth)):
                if (p / "train.csv").is_file() and (p / "train_series").is_dir():
                    return p
    raise SystemExit(
        "No encuentro los datos. En Kaggle: Add Data -> Competition 'rsna-knee-abnormality-detection'. "
        "En local: export P1_DATA_ROOT=/ruta/a/los/datos"
    )


def _work_root() -> Path:
    env = os.environ.get("P1_WORK")
    if env:
        return Path(env)
    if Path("/kaggle/working").is_dir():
        return Path("/kaggle/working/p1")
    return Path(__file__).resolve().parents[1]  # la carpeta P1_RSNA_rodilla


WORK = _work_root()
DIR_DATOS = WORK / "05_datos"
DIR_RES = WORK / "06_resultados"
DIR_FIG = DIR_RES / "figuras"
DIR_LOGS = WORK / "02_logs"
DIR_RUNS = DIR_LOGS / "runs"
DIR_CACHE = Path(os.environ.get("P1_CACHE", str(WORK / "cache")))
for _d in (DIR_DATOS, DIR_RES, DIR_FIG, DIR_LOGS, DIR_RUNS, DIR_CACHE):
    _d.mkdir(parents=True, exist_ok=True)


def data_root() -> Path:
    return _find_data_root()


# ----------------------------------------------------------------------------------------------
# Hiperparámetros por defecto (ver 01_decisiones/D004-D007)
# ----------------------------------------------------------------------------------------------
N_SUBSET = int(os.environ.get("P1_N_SUBSET", 500))    # estudios débiles a usar (300-600 sugerido)
MIN_POS_PER_LABEL = int(os.environ.get("P1_MIN_POS", 25))                                 # positivos mínimos por hallazgo en el subconjunto
N_SLICES = 9          # cortes por serie (2.5D): muestreados uniformemente en el 80 % central
IMG_SIZE = 256        # tamaño de cache (px); entrenamiento hace random-crop a 224
CROP_SIZE = 224
SLOTS = {             # plano -> filtro sobre train_series.csv
    "sag_fluid": ("Sagittal", 1),   # principal: sagital PD/T2 con supresión grasa
    "cor_fluid": ("Coronal", 1),    # ablación de plano
}


# ----------------------------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------------------------
def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def now() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Logger:
    """Escribe a consola y a 02_logs/<nombre>.log (append) para dejar trazabilidad."""

    def __init__(self, name: str):
        self.path = DIR_LOGS / f"{name}.log"
        self.f = open(self.path, "a", encoding="utf-8")
        self(f"===== inicio {name} | {now()} | python {sys.version.split()[0]} =====")

    def __call__(self, *msg):
        line = " ".join(str(m) for m in msg)
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def append_bitacora(texto: str) -> None:
    """Agrega una línea fechada a 02_logs/bitacora_ejecuciones.md."""
    with open(DIR_LOGS / "bitacora_ejecuciones.md", "a", encoding="utf-8") as f:
        f.write(f"- {now()} — {texto}\n")


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)
