"""08 — Consolida todos los runs en tablas y figuras listas para el informe (Entregables 3, 4, 6).

Lee 02_logs/runs/*/metricas.json y pred_dorado.csv y produce:
  06_resultados/08_comparacion_runs.csv / .md / .tex   (una fila por run)
  06_resultados/08_auc_por_hallazgo.csv / .tex          (hallazgo x run)
  06_resultados/08_ablaciones_pareadas.csv              (Δ AUC vs base con IC 95 % bootstrap pareado)
  06_resultados/08_desbalance.csv                       (prevalencia + AUC en raros con/sin mitigación)
  06_resultados/figuras/08_auc_heatmap.png, 08_curvas_entrenamiento.png

Uso: python 08_tablas_para_informe.py [--base A2_resnet50]
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config as C  # noqa: E402
import metricas as L  # noqa: E402

log = C.Logger("08_tablas_para_informe")


def load_preds(run):
    d = pd.read_csv(C.DIR_RUNS / run / "pred_dorado.csv").sort_values(C.STUDY_COL)
    return d[C.STUDY_COL].values, d[[f"y::{l}" for l in C.LABELS]].values, d[[f"p::{l}" for l in C.LABELS]].values


def paired_delta(run_a, run_b, n=2000):
    ia, Ya, Pa = load_preds(run_a)
    ib, Yb, Pb = load_preds(run_b)
    common = np.intersect1d(ia, ib)
    ma, mb = np.isin(ia, common), np.isin(ib, common)
    Y, Pa, Pb = Ya[ma], Pa[ma], Pb[mb]
    rng = np.random.default_rng(C.SEED)
    d = []
    for _ in range(n):
        idx = rng.integers(0, len(Y), len(Y))
        d.append(L.macro_auc(Y[idx], Pb[idx]) - L.macro_auc(Y[idx], Pa[idx]))
    obs = L.macro_auc(Y, Pb) - L.macro_auc(Y, Pa)
    return obs, float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5)), float(np.mean(np.array(d) > 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="A2_resnet50")
    args = ap.parse_args()
    runs = sorted(p.parent.name for p in C.DIR_RUNS.glob("*/metricas.json"))
    if not runs:
        raise SystemExit("No hay runs con metricas.json todavía")
    M = [json.load(open(C.DIR_RUNS / r / "metricas.json")) for r in runs]
    cols = ["run", "arch", "slot", "loss", "ruido", "pos_weight", "sampler", "parametros_M", "mejor_epoca",
            "auc_val_debil", "auc_macro_dorado", "ic95_dorado", "f1_macro_dorado", "precision_macro_dorado",
            "recall_macro_dorado", "min_entrenamiento"]
    comp = pd.DataFrame(M)[cols]
    comp["ic95_dorado"] = comp["ic95_dorado"].map(lambda v: f"[{v[0]:.3f}, {v[1]:.3f}]")
    comp = comp.round(4)
    comp.to_csv(C.DIR_RES / "08_comparacion_runs.csv", index=False)
    try:
        (C.DIR_RES / "08_comparacion_runs.md").write_text(comp.to_markdown(index=False), encoding="utf-8")
        comp.to_latex(C.DIR_RES / "08_comparacion_runs.tex", index=False, escape=True)
    except ImportError as e:  # tabulate / jinja2 ausentes
        log("md/tex no generados:", e)
    log("\n" + comp.to_string(index=False))

    auc = pd.DataFrame({m["run"]: m["auc_por_hallazgo"] for m in M}).loc[C.LABELS]
    auc.loc["MACRO"] = auc.mean()
    auc.round(3).to_csv(C.DIR_RES / "08_auc_por_hallazgo.csv")
    try:
        auc.round(3).to_latex(C.DIR_RES / "08_auc_por_hallazgo.tex", escape=True)
    except ImportError:
        pass

    fig, ax = plt.subplots(figsize=(1.2 * len(runs) + 3, 6))
    im = ax.imshow(auc.values, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(runs)), runs, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(auc.index)), [C.LABELS_ES.get(i, i) for i in auc.index], fontsize=8)
    for i in range(auc.shape[0]):
        for j in range(auc.shape[1]):
            v = auc.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, label="AUC (dorados)")
    ax.set_title("AUC por hallazgo en el conjunto dorado")
    fig.tight_layout()
    fig.savefig(C.DIR_FIG / "08_auc_heatmap.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4))
    for r in runs:
        h = pd.read_csv(C.DIR_RUNS / r / "historial.csv")
        ax.plot(h["epoca"], h["auc_val_debil"], marker=".", label=r)
    ax.set_xlabel("época")
    ax.set_ylabel("AUC macro (val, etiquetas débiles)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(C.DIR_FIG / "08_curvas_entrenamiento.png", dpi=150)

    if args.base in runs:
        filas = []
        for r in runs:
            if r == args.base:
                continue
            obs, lo, hi, p = paired_delta(args.base, r)
            filas.append({"comparacion": f"{r} vs {args.base}", "delta_auc": obs, "ic95_lo": lo, "ic95_hi": hi,
                          "P(delta>0)": p})
        pdlt = pd.DataFrame(filas).round(4)
        pdlt.to_csv(C.DIR_RES / "08_ablaciones_pareadas.csv", index=False)
        log("\n" + pdlt.to_string(index=False))

    prev_p = C.DIR_RES / "04_prevalencia.csv"
    if prev_p.exists():
        prev = pd.read_csv(prev_p).set_index("hallazgo")
        des = prev.copy()
        for r in runs:
            des[f"auc_{r}"] = auc.loc[C.LABELS, r]
        des = des.sort_values("prev_train")
        des.round(3).to_csv(C.DIR_RES / "08_desbalance.csv")
        log("\n" + des.round(3).to_string())
    C.append_bitacora(f"08 tablas consolidadas: {len(runs)} runs")


if __name__ == "__main__":
    main()
