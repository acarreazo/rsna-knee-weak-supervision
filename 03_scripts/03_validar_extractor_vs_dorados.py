"""03 — Validación del extractor de etiquetas débiles contra los ~58 estudios dorados (Entregable 1).

Para cada método disponible (reglas, NLI, ensamble = promedio de rangos) reporta por hallazgo:
  AUC (puntaje ordinal vs dorado), acuerdo (accuracy), sensibilidad, especificidad, kappa de Cohen,
  y la TASA DE RUIDO estimada = 1 - acuerdo. Además: acuerdo por idioma (insumo del Entregable 7)
  e IC 95 % bootstrap del AUC macro.

Salidas: 06_resultados/03_validacion_extractor.csv, 03_acuerdo_por_idioma.csv, 03_resumen.json
         06_resultados/figuras/03_extractor_auc_por_hallazgo.png
"""
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

import config as C

log = C.Logger("03_validar_extractor")


def safe_auc(y, s):
    y = np.asarray(y)
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def evaluar(gold: pd.DataFrame, scores: pd.DataFrame, nombre: str, thr: float = 0.5):
    filas = []
    for lab in C.LABELS:
        y = gold[lab].astype(int).values
        s = scores[f"score::{lab}"].values
        b = (s >= thr).astype(int)
        tp, tn = int(((b == 1) & (y == 1)).sum()), int(((b == 0) & (y == 0)).sum())
        fp, fn = int(((b == 1) & (y == 0)).sum()), int(((b == 0) & (y == 1)).sum())
        filas.append({
            "metodo": nombre, "hallazgo": lab, "n_pos_dorado": int(y.sum()),
            "auc": safe_auc(y, s), "acuerdo": (tp + tn) / len(y),
            "sensibilidad": tp / max(tp + fn, 1), "especificidad": tn / max(tn + fp, 1),
            "kappa": cohen_kappa_score(y, b) if len(set(b)) > 1 or len(set(y)) > 1 else np.nan,
            "tasa_ruido": 1 - (tp + tn) / len(y),
        })
    return pd.DataFrame(filas)


def bootstrap_macro_auc(gold, scores, n=1000, seed=C.SEED):
    rng = np.random.default_rng(seed)
    Y = gold[C.LABELS].astype(int).values
    S = scores[[f"score::{l}" for l in C.LABELS]].values
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(Y), len(Y))
        aucs = [safe_auc(Y[idx, j], S[idx, j]) for j in range(C.N_LABELS)]
        vals.append(np.nanmean(aucs))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    root = C.data_root()
    tr = pd.read_csv(root / "train.csv")
    gold = tr[tr[C.LABELS].notna().all(axis=1)][[C.STUDY_COL] + C.LABELS].reset_index(drop=True)
    est = pd.read_csv(C.DIR_DATOS / "estudios.csv")[[C.STUDY_COL, "idioma"]]
    gold = gold.merge(est, on=C.STUDY_COL, how="left")
    log(f"estudios dorados: {len(gold)}")

    metodos = {}
    reglas = pd.read_csv(C.DIR_DATOS / "etiquetas_debiles_reglas.csv")
    metodos["reglas"] = gold[[C.STUDY_COL]].merge(reglas, on=C.STUDY_COL, how="left")
    nli_path = C.DIR_DATOS / "etiquetas_debiles_nli.csv"
    if nli_path.exists():
        nli = pd.read_csv(nli_path)
        metodos["nli"] = gold[[C.STUDY_COL]].merge(nli, on=C.STUDY_COL, how="left").fillna(0.0)
        ens = metodos["reglas"][[C.STUDY_COL]].copy()
        for lab in C.LABELS:
            r1 = metodos["reglas"][f"score::{lab}"].rank(pct=True)
            r2 = metodos["nli"][f"score::{lab}"].rank(pct=True)
            ens[f"score::{lab}"] = (r1 + r2) / 2
        metodos["ensamble"] = ens

    tablas, resumen = [], {}
    for nombre, sc in metodos.items():
        thr = 0.5
        if nombre == "ensamble":
            thr = float(np.median([sc[f"score::{l}"].quantile(1 - gold[l].mean()) for l in C.LABELS]))
        t = evaluar(gold, sc, nombre, thr)
        lo, hi = bootstrap_macro_auc(gold, sc)
        resumen[nombre] = {
            "auc_macro": float(t["auc"].mean()), "ic95_auc_macro": [lo, hi],
            "acuerdo_medio": float(t["acuerdo"].mean()), "tasa_ruido_media": float(t["tasa_ruido"].mean()),
            "kappa_medio": float(t["kappa"].mean()),
        }
        log(f"[{nombre}] AUC macro={t['auc'].mean():.3f} IC95=[{lo:.3f},{hi:.3f}] "
            f"acuerdo={t['acuerdo'].mean():.3f} ruido={t['tasa_ruido'].mean():.3f}")
        log(t.round(3).to_string(index=False))
        tablas.append(t)
    tab = pd.concat(tablas)
    tab.to_csv(C.DIR_RES / "03_validacion_extractor.csv", index=False)

    # acuerdo por idioma (método de reglas) -> discusión de sesgo lingüístico
    sc = metodos["reglas"]
    filas = []
    for lang, g in gold.groupby("idioma"):
        idx = g.index
        Y = g[C.LABELS].astype(int).values
        B = sc.loc[idx, [f"weak::{l}" for l in C.LABELS]].values
        filas.append({"idioma": lang, "n_estudios": len(g), "acuerdo": float((Y == B).mean()),
                      "sensibilidad": float(((Y == 1) & (B == 1)).sum() / max((Y == 1).sum(), 1))})
    por_idioma = pd.DataFrame(filas).sort_values("n_estudios", ascending=False)
    por_idioma.to_csv(C.DIR_RES / "03_acuerdo_por_idioma.csv", index=False)
    log("acuerdo por idioma:\n" + por_idioma.round(3).to_string(index=False))
    resumen["acuerdo_por_idioma"] = por_idioma.to_dict(orient="records")
    C.save_json(resumen, C.DIR_RES / "03_resumen.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        piv = tab.pivot(index="hallazgo", columns="metodo", values="auc").loc[C.LABELS]
        ax = piv.plot.barh(figsize=(7, 5))
        ax.axvline(0.5, color="gray", ls="--", lw=1)
        ax.set_xlabel("AUC vs etiqueta dorada (n≈58)")
        ax.set_title("Extractor de etiquetas débiles por hallazgo")
        plt.tight_layout()
        plt.savefig(C.DIR_FIG / "03_extractor_auc_por_hallazgo.png", dpi=150)
    except Exception as e:  # noqa: BLE001
        log("figura no generada:", e)
    C.append_bitacora("03 validación extractor: " + ", ".join(
        f"{k} AUC={v['auc_macro']:.3f} ruido={v['tasa_ruido_media']:.3f}" for k, v in resumen.items()
        if isinstance(v, dict) and "auc_macro" in v))


if __name__ == "__main__":
    main()
