"""07 — Interpretabilidad: atención por corte + Grad-CAM (Entregable 5).

Para cada hallazgo elegido toma los verdaderos positivos con mayor probabilidad (dorados primero,
luego validación), identifica el corte con mayor atención para ese hallazgo y calcula Grad-CAM
sobre la última capa convolucional del encoder en ese corte.

Recordatorio para la discusión: los cortes sagitales están ordenados MEDIAL -> LATERAL
(posición 0 = más medial). Un mapa plausible de "Medial Meniscus" debería salir en cortes
de índice bajo y resaltar el cuerno posterior/anterior; "ACL" en cortes centrales (escotadura
intercondílea); "Effusion" en el receso suprapatelar; "Baker's" en la región poplítea posterior.

Uso:  python 07_gradcam.py --run A2_resnet50 [--hallazgos "ACL,Medial Meniscus,Effusion,Baker's"] [--n 2]
Salida: 06_resultados/figuras/07_gradcam_<run>.png, 07_atencion_<run>.png, 07_gradcam_<run>.csv
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import config as C  # noqa: E402
import p1lib as L  # noqa: E402


def gradcam(model, vol_u8, label_idx, dev):
    acts, grads = {}, {}
    layer = model.encoder.cam_layer
    def fwd_hook(m, i, o):
        acts["a"] = o
        o.register_hook(lambda g: grads.__setitem__("g", g))

    h1 = layer.register_forward_hook(fwd_hook)
    x = L.gpu_augment(torch.from_numpy(vol_u8)[None].to(dev), train=False)
    model.zero_grad()
    logits, att = model(x, return_att=True)
    logits[0, label_idx].backward()
    h1.remove()
    a, g = acts["a"].detach(), grads["g"].detach()          # (S, C, h, w)
    att = att[0, :, label_idx].detach().cpu().numpy()       # (S,)
    s = int(att.argmax())
    w = g[s].mean(dim=(1, 2), keepdim=True)
    cam = F.relu((w * a[s]).sum(0))
    cam = F.interpolate(cam[None, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return x[0, s].cpu().numpy(), cam.cpu().numpy(), s, att, torch.sigmoid(logits[0, label_idx]).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--hallazgos", default="ACL,Medial Meniscus,Effusion,Baker's,Contusion")
    ap.add_argument("--n", type=int, default=2)
    args = ap.parse_args()
    log = C.Logger("07_gradcam")
    run_dir = C.DIR_RUNS / args.run
    cfg = json.load(open(run_dir / "config.json"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = L.KneeNet(cfg["arch"], pretrained=False).to(dev)
    model.load_state_dict(torch.load(run_dir / "mejor.pt", map_location=dev))
    model.eval()

    preds = pd.concat([pd.read_csv(run_dir / "pred_dorado.csv").assign(origen="dorado"),
                       pd.read_csv(run_dir / "pred_val.csv").assign(origen="val")])
    labs = [l.strip() for l in args.hallazgos.split(",")]
    fig, axes = plt.subplots(len(labs), 2 * args.n, figsize=(3.2 * 2 * args.n, 3.2 * len(labs)))
    axes = np.atleast_2d(axes)
    fig_a, ax_a = plt.subplots(figsize=(7, 4))
    filas = []
    for r, lab in enumerate(labs):
        j = C.LABELS.index(lab)
        tp = preds[preds[f"y::{lab}"] == 1].sort_values(["origen", f"p::{lab}"], ascending=[True, False])
        tp = tp.head(args.n)
        for k in range(2 * args.n):
            axes[r, k].axis("off")
        for c, (_, row) in enumerate(tp.iterrows()):
            uid = row[C.STUDY_COL]
            vol = np.load(C.DIR_CACHE / cfg["slot"] / f"{uid}.npy")
            img, cam, s, att, p = gradcam(model, vol, j, dev)
            axes[r, 2 * c].imshow(img, cmap="gray")
            axes[r, 2 * c].set_title(f"{C.LABELS_ES[lab]}\ncorte {s}/{len(att) - 1} ({row['origen']})", fontsize=8)
            axes[r, 2 * c + 1].imshow(img, cmap="gray")
            axes[r, 2 * c + 1].imshow(cam, cmap="jet", alpha=0.4)
            axes[r, 2 * c + 1].set_title(f"Grad-CAM p={p:.2f}", fontsize=8)
            if c == 0:
                ax_a.plot(att, marker="o", label=C.LABELS_ES[lab])
            filas.append({"hallazgo": lab, C.STUDY_COL: uid, "origen": row["origen"], "prob": p,
                          "corte_max_atencion": s, "atencion": np.round(att, 3).tolist()})
    fig.suptitle(f"Grad-CAM — {args.run} (cortes sagitales ordenados medial→lateral)")
    fig.tight_layout()
    fig.savefig(C.DIR_FIG / f"07_gradcam_{args.run}.png", dpi=130)
    ax_a.set_xlabel("índice de corte (0 = medial → lateral)")
    ax_a.set_ylabel("atención")
    ax_a.legend(fontsize=7)
    ax_a.set_title("Atención por corte (primer caso de cada hallazgo)")
    fig_a.tight_layout()
    fig_a.savefig(C.DIR_FIG / f"07_atencion_{args.run}.png", dpi=130)
    pd.DataFrame(filas).to_csv(C.DIR_RES / f"07_gradcam_{args.run}.csv", index=False)
    log(pd.DataFrame(filas)[["hallazgo", "origen", "prob", "corte_max_atencion"]].to_string(index=False))
    C.append_bitacora(f"07 Grad-CAM {args.run}: {len(filas)} casos en {len(labs)} hallazgos")


if __name__ == "__main__":
    main()
