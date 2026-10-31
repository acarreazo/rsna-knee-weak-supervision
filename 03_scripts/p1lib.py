"""Código compartido: dataset 2.5D, arquitecturas, pérdidas y métricas.

Arquitectura común (ver D006): un ENCODER 2D compartido procesa cada corte (con sus vecinos como
canales RGB -> "2.5D"), y una capa de ATENCIÓN POR HALLAZGO agrega los cortes del estudio:
    x: (B, S, H, W) -> por corte s: [s-1, s, s+1] como 3 canales -> encoder -> h (B, S, F)
    a_l = softmax_s( w_l^T tanh(V h_s) )      (atención de cada hallazgo l sobre los S cortes)
    z_l = sum_s a_l,s h_s ;  logit_l = u_l^T z_l + b_l        -> 12 salidas sigmoides
La atención indica QUÉ corte usó el modelo para cada hallazgo y Grad-CAM indica DÓNDE dentro
del corte -> interpretabilidad en dos niveles.

Encoders:  scratch (CNN propia, 4 bloques)  |  resnet18 / resnet50 / efficientnet_b0 (ImageNet)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C

IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


# =============================================================================================
# Datos
# =============================================================================================
class KneeDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, slot: str, target: str = "binario"):
        self.df = df.reset_index(drop=True)
        self.slot = slot
        if target == "suave":
            self.y = self.df[[f"score::{l}" for l in C.LABELS]].values.astype(np.float32)
            # en dorados el score es el de reglas; se usa la etiqueta dorada para evaluar
            gold = (self.df["split"] == "test_dorado").values
            self.y[gold] = self.df.loc[gold, [f"weak::{l}" for l in C.LABELS]].values
        else:
            self.y = self.df[[f"weak::{l}" for l in C.LABELS]].values.astype(np.float32)
        self.w = self.df[[f"conf::{l}" for l in C.LABELS]].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        uid = self.df.loc[i, C.STUDY_COL]
        vol = np.load(C.DIR_CACHE / self.slot / f"{uid}.npy")  # (S, H, W) uint8
        return torch.from_numpy(vol), torch.from_numpy(self.y[i]), torch.from_numpy(self.w[i]), i


def gpu_augment(x: torch.Tensor, train: bool, crop: int = C.CROP_SIZE) -> torch.Tensor:
    """x uint8 (B,S,H,W) -> float (B,S,crop,crop). Sin flip horizontal (invierte medial/lateral)."""
    x = x.float() / 255.0
    B, S, H, W = x.shape
    if train:
        out = []
        for b in range(B):
            r0 = np.random.randint(0, H - crop + 1)
            c0 = np.random.randint(0, W - crop + 1)
            out.append(x[b, :, r0:r0 + crop, c0:c0 + crop])
        x = torch.stack(out)
        # rotación pequeña (±10°) y escala (0.9-1.1), misma para todos los cortes del estudio
        ang = (torch.rand(B, device=x.device) - 0.5) * (20 * np.pi / 180)
        sc = 0.9 + 0.2 * torch.rand(B, device=x.device)
        cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
        theta = torch.stack([torch.stack([cos, -sin, torch.zeros_like(cos)], 1),
                             torch.stack([sin, cos, torch.zeros_like(cos)], 1)], 1)
        grid = F.affine_grid(theta, (B, S, crop, crop), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
        # brillo / contraste
        g = 0.85 + 0.3 * torch.rand(B, 1, 1, 1, device=x.device)
        bshift = (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * 0.1
        x = (x * g + bshift).clamp(0, 1)
    else:
        r0, c0 = (H - crop) // 2, (W - crop) // 2
        x = x[:, :, r0:r0 + crop, c0:c0 + crop]
    return x


def to_25d(x: torch.Tensor) -> torch.Tensor:
    """(B,S,H,W) -> (B,S,3,H,W) con [s-1, s, s+1] como canales (bordes replicados)."""
    prev = torch.cat([x[:, :1], x[:, :-1]], 1)
    nxt = torch.cat([x[:, 1:], x[:, -1:]], 1)
    return torch.stack([prev, x, nxt], 2)


# =============================================================================================
# Modelos
# =============================================================================================
class ScratchCNN(nn.Module):
    """CNN 2D diseñada desde cero: 5 bloques (conv-BN-ReLU) x2 + maxpool, GAP. ~1.9 M parámetros."""

    def __init__(self, chans=(32, 64, 128, 192, 256)):
        super().__init__()
        layers, cin = [], 3
        for c in chans:
            layers += [nn.Conv2d(cin, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(),
                       nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(),
                       nn.MaxPool2d(2)]
            cin = c
        self.features = nn.Sequential(*layers)
        self.out_dim = cin

    def forward(self, x):
        return F.adaptive_avg_pool2d(self.features(x), 1).flatten(1)

    @property
    def cam_layer(self):
        return self.features[-2]  # última ReLU antes del último pool


class TVBackbone(nn.Module):
    def __init__(self, name: str, pretrained: bool = True):
        super().__init__()
        import torchvision.models as tvm
        if name in ("resnet18", "resnet50"):
            w = {"resnet18": tvm.ResNet18_Weights.IMAGENET1K_V1,
                 "resnet50": tvm.ResNet50_Weights.IMAGENET1K_V2}[name] if pretrained else None
            m = getattr(tvm, name)(weights=w)
            self.out_dim = m.fc.in_features
            m.fc = nn.Identity()
            self.net, self._cam = m, m.layer4
        elif name == "efficientnet_b0":
            w = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            m = tvm.efficientnet_b0(weights=w)
            self.out_dim = m.classifier[1].in_features
            m.classifier = nn.Identity()
            self.net, self._cam = m, m.features[-1]
        else:
            raise ValueError(name)

    def forward(self, x):
        return self.net(x)

    @property
    def cam_layer(self):
        return self._cam


class KneeNet(nn.Module):
    def __init__(self, arch: str = "resnet50", pretrained: bool = True, att_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.arch = arch
        self.imagenet = arch != "scratch"
        self.encoder = ScratchCNN() if arch == "scratch" else TVBackbone(arch, pretrained)
        Fd = self.encoder.out_dim
        self.V = nn.Linear(Fd, att_dim)
        self.att = nn.Linear(att_dim, C.N_LABELS)
        self.drop = nn.Dropout(dropout)
        self.U = nn.Parameter(torch.randn(C.N_LABELS, Fd) * 0.01)
        self.b = nn.Parameter(torch.zeros(C.N_LABELS))

    def forward(self, x, return_att: bool = False):
        # x: (B,S,H,W) float en [0,1]
        x = to_25d(x)
        if self.imagenet:
            x = (x - IMNET_MEAN.to(x.device)) / IMNET_STD.to(x.device)
        else:
            x = (x - 0.5) / 0.25
        B, S = x.shape[:2]
        h = self.encoder(x.flatten(0, 1)).view(B, S, -1)            # (B,S,F)
        a = torch.softmax(self.att(torch.tanh(self.V(h))), dim=1)    # (B,S,L)
        z = torch.einsum("bsl,bsf->blf", a, h)                       # (B,L,F)
        logits = (self.drop(z) * self.U).sum(-1) + self.b            # (B,L)
        return (logits, a) if return_att else logits


def param_groups(model: KneeNet, lr: float, lr_backbone: float):
    enc = list(model.encoder.parameters())
    ids = {id(p) for p in enc}
    head = [p for p in model.parameters() if id(p) not in ids]
    return [{"params": enc, "lr": lr_backbone}, {"params": head, "lr": lr}]


# =============================================================================================
# Pérdidas
# =============================================================================================
def loss_fn(logits, y, w, kind: str = "bce", pos_weight=None, gamma: float = 2.0):
    """Pérdida por elemento ponderada (w = confianza o máscara) y promediada."""
    if kind == "bce":
        l = F.binary_cross_entropy_with_logits(logits, y, reduction="none", pos_weight=pos_weight)
    elif kind == "focal":
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none", pos_weight=pos_weight)
        p = torch.sigmoid(logits)
        pt = p * y + (1 - p) * (1 - y)
        l = (1 - pt) ** gamma * bce
    else:
        raise ValueError(kind)
    return (l * w).sum() / w.sum().clamp_min(1.0)


# Métricas: en metricas.py (sin dependencia de torch)
from metricas import best_thresholds, bootstrap_ci, macro_auc, per_label_report, safe_auc  # noqa: E402,F401
