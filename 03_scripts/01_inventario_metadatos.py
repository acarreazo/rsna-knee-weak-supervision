"""01 — Inventario de metadatos (sin abrir imágenes).

Lee train.csv y train_series.csv, identifica los ~58 estudios con etiqueta dorada, detecta el
idioma de cada reporte y resume la disponibilidad de planos/secuencias.

Salidas:
  05_datos/estudios.csv            una fila por estudio: uid, idioma, n_palabras, es_dorado, slots
  06_resultados/01_inventario.json resumen numérico (para el informe)
"""
import re
import unicodedata

import numpy as np
import pandas as pd

import config as C

log = C.Logger("01_inventario_metadatos")

# ---------------------------------------------------------------------------------------------
# Detección de idioma por guion + palabras vacías (sin dependencias externas)
# ---------------------------------------------------------------------------------------------
STOP = {
    "en": "the and of with is no there are without normal seen".split(),
    "de": "der die und mit ist kein keine nicht im des ohne bei".split(),
    "es": "el la de los las con sin se del en por que".split(),
    "pt": "do da com sem dos das não em que se ao".split(),
    "fr": "le la les des du avec sans pas une est au".split(),
    "nl": "de het een van met geen niet en is zijn op".split(),
    "it": "il di della con senza non del delle gli si".split(),
    "tr": "ve ile bir yok olarak izlendi mevcut saptanmadı".split(),
}


def detect_lang(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return "sin_reporte"
    if re.search(r"[Ѐ-ӿ]", text):
        return "ru"
    if re.search(r"[Ͱ-Ͽ]", text):
        return "el"
    if re.search(r"[一-鿿]", text):
        return "zh"
    if re.search(r"[぀-ヿ]", text):
        return "ja"
    if re.search(r"[가-힯]", text):
        return "ko"
    if re.search(r"[ğışĞİŞ]", text):
        return "tr"
    toks = re.findall(r"[^\W\d_]+", text.lower())
    if not toks:
        return "otro"
    best, best_sc = "otro", 0.0
    for lang, words in STOP.items():
        sc = sum(t in words for t in toks) / len(toks)
        if sc > best_sc:
            best, best_sc = lang, sc
    return best if best_sc > 0.02 else "otro"


def main():
    root = C.data_root()
    log("datos en:", root)
    tr = pd.read_csv(root / "train.csv")
    ts = pd.read_csv(root / "train_series.csv")
    log("train.csv", tr.shape, "columnas:", list(tr.columns))
    log("train_series.csv", ts.shape, "columnas:", list(ts.columns))

    faltan = [l for l in C.LABELS if l not in tr.columns]
    if faltan:
        raise SystemExit(f"Columnas de etiqueta no encontradas en train.csv: {faltan}. Revisar config.LABELS")

    n_lab = tr[C.LABELS].notna().sum(axis=1)
    tr["es_dorado"] = n_lab == C.N_LABELS
    rep = tr[C.REPORT_COL].fillna("").astype(str)
    tr["idioma"] = rep.map(detect_lang)
    tr["n_palabras"] = rep.str.split().str.len().fillna(0).astype(int)

    # slots disponibles por estudio
    for slot, (plane, fluid) in C.SLOTS.items():
        m = (ts[C.PLANE_COL] == plane) & (ts[C.FLUID_COL] == fluid)
        tr[f"tiene_{slot}"] = tr[C.STUDY_COL].isin(set(ts.loc[m, C.STUDY_COL]))

    cols = [C.STUDY_COL, "idioma", "n_palabras", "es_dorado"] + [f"tiene_{s}" for s in C.SLOTS]
    tr[cols].to_csv(C.DIR_DATOS / "estudios.csv", index=False)

    gold = tr[tr.es_dorado]
    prev_gold = gold[C.LABELS].mean().round(3).to_dict()
    combos = (ts[C.PLANE_COL].astype(str) + "|fluid=" + ts[C.FLUID_COL].astype(str)).value_counts()

    resumen = {
        "n_estudios": int(len(tr)),
        "n_series": int(len(ts)),
        "n_dorados": int(tr.es_dorado.sum()),
        "n_con_reporte": int((rep.str.strip() != "").sum()),
        "idiomas": tr["idioma"].value_counts().to_dict(),
        "idiomas_dorados": gold["idioma"].value_counts().to_dict(),
        "prevalencia_dorados": prev_gold,
        "palabras_reporte_p50": float(tr["n_palabras"].median()),
        "series_por_combo": combos.to_dict(),
        "cobertura_slots": {s: float(tr[f"tiene_{s}"].mean()) for s in C.SLOTS},
    }
    C.save_json(resumen, C.DIR_RES / "01_inventario.json")
    for k, v in resumen.items():
        log(f"{k}: {v}")
    C.append_bitacora(
        f"01 inventario: {resumen['n_estudios']} estudios, {resumen['n_dorados']} dorados, "
        f"{len(resumen['idiomas'])} idiomas; cobertura sag_fluid={resumen['cobertura_slots'].get('sag_fluid', 0):.1%}"
    )


if __name__ == "__main__":
    main()
