"""04 — Selección del subconjunto estratificado (Entregables 6 y "reducción de escala").

Reglas (ver D004):
  * Los estudios dorados (~58) quedan SOLO como conjunto de prueba limpio ("test_dorado").
  * Pool de entrenamiento = estudios no dorados con reporte y con serie sagital sensible a líquido
    (y coronal sensible a líquido, para poder hacer la ablación de plano con los mismos estudios).
  * Muestreo estratificado: recorremos los hallazgos de menor a mayor prevalencia débil y
    tomamos hasta MIN_POS_PER_LABEL positivos de cada uno; luego completamos al azar hasta N.
  * División train / val (85 / 15) estratificada por el hallazgo positivo más raro de cada estudio.
    'val' (etiquetas débiles) se usa para early stopping y para elegir umbrales; los dorados
    NUNCA se usan para decidir nada (evita sobreajustar a 58 casos).

Salidas: 05_datos/subconjunto.csv (uid, split, etiquetas débiles, confianzas)
         06_resultados/04_prevalencia.csv  (prevalencia por hallazgo en train/val/dorados)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config as C

log = C.Logger("04_seleccion_subconjunto")


def main():
    C.seed_everything()
    root = C.data_root()
    tr = pd.read_csv(root / "train.csv")
    est = pd.read_csv(C.DIR_DATOS / "estudios.csv")
    weak = pd.read_csv(C.DIR_DATOS / "etiquetas_debiles_reglas.csv")
    df = est.merge(weak, on=C.STUDY_COL)

    gold = df[df.es_dorado].copy()
    pool = df[(~df.es_dorado) & df.tiene_sag_fluid & df.tiene_cor_fluid & (df.n_palabras > 0)].copy()
    log(f"pool elegible: {len(pool)} | dorados: {len(gold)} (con sag_fluid: {int(gold.tiene_sag_fluid.sum())})")

    rng = np.random.default_rng(C.SEED)
    wcols = [f"weak::{l}" for l in C.LABELS]
    prev = pool[wcols].mean().sort_values()
    elegidos = set()
    for col in prev.index:
        cand = pool[(pool[col] == 1) & (~pool[C.STUDY_COL].isin(elegidos))][C.STUDY_COL].values
        ya = pool[pool[C.STUDY_COL].isin(elegidos)][col].sum()
        k = int(max(C.MIN_POS_PER_LABEL - ya, 0))
        if k and len(cand):
            elegidos.update(rng.choice(cand, size=min(k, len(cand)), replace=False))
    resto = pool[~pool[C.STUDY_COL].isin(elegidos)][C.STUDY_COL].values
    falta = max(C.N_SUBSET - len(elegidos), 0)
    elegidos.update(rng.choice(resto, size=min(falta, len(resto)), replace=False))
    sub = pool[pool[C.STUDY_COL].isin(elegidos)].copy()

    # estrato = hallazgo positivo más raro del estudio
    orden = {c: i for i, c in enumerate(prev.index)}

    def estrato(r):
        pos = [c for c in prev.index if r[c] == 1]
        return min(pos, key=orden.get) if pos else "ninguno"

    sub["estrato"] = sub.apply(estrato, axis=1)
    vc = sub["estrato"].value_counts()
    sub.loc[sub["estrato"].isin(vc[vc < 4].index), "estrato"] = "otros"
    tr_ids, va_ids = train_test_split(sub[C.STUDY_COL], test_size=0.15, random_state=C.SEED, stratify=sub["estrato"])
    sub["split"] = np.where(sub[C.STUDY_COL].isin(set(va_ids)), "val", "train")

    gold = gold[gold.tiene_sag_fluid].copy()
    gold["split"] = "test_dorado"
    gold = gold.drop(columns=wcols).merge(tr[[C.STUDY_COL] + C.LABELS], on=C.STUDY_COL)
    for l in C.LABELS:   # para dorados, la "etiqueta" de evaluación es la dorada
        gold[f"weak::{l}"] = gold[l].astype(int)
        gold[f"conf::{l}"] = 1.0
    keep = [C.STUDY_COL, "split", "idioma", "n_palabras"] + wcols + [f"conf::{l}" for l in C.LABELS] + \
           [f"score::{l}" for l in C.LABELS]
    final = pd.concat([sub[keep], gold[keep]], ignore_index=True)
    final.to_csv(C.DIR_DATOS / "subconjunto.csv", index=False)

    tab = final.groupby("split")[wcols].agg(["sum", "mean"])
    filas = []
    for l in C.LABELS:
        fila = {"hallazgo": l}
        for s in ["train", "val", "test_dorado"]:
            g = final[final.split == s][f"weak::{l}"]
            fila[f"n_pos_{s}"] = int(g.sum())
            fila[f"prev_{s}"] = round(float(g.mean()), 3) if len(g) else np.nan
        filas.append(fila)
    prev_tab = pd.DataFrame(filas)
    prev_tab.to_csv(C.DIR_RES / "04_prevalencia.csv", index=False)
    log(final["split"].value_counts().to_string())
    log("prevalencia por split:\n" + prev_tab.to_string(index=False))
    C.append_bitacora(f"04 subconjunto: {final.split.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
