"""02b — (Opcional) Extractor de etiquetas débiles con un modelo multilingüe de NLI (zero-shot).

Segunda opción que pide el enunciado ("embeddings multilingües con clasificación de texto de
bajo costo"). Usa un modelo de pesos abiertos que corre DENTRO del notebook de Kaggle: el texto
de los reportes NO se envía a ninguna API externa (regla de la competencia; ver D003).

Para cada reporte y hallazgo se evalúa la hipótesis en inglés (p.ej. "The anterior cruciate
ligament is torn.") y se toma P(entailment) como puntaje. Requiere GPU e Internet ON (para
descargar el modelo la primera vez). Solo se procesan los estudios que usaremos (dorados +
subconjunto), así el costo es de minutos.

Salida: 05_datos/etiquetas_debiles_nli.csv  (columnas score::<label>)
Uso:    python 02b_extractor_nli_multilingue.py [--todos]
"""
import argparse

import numpy as np
import pandas as pd

import config as C

log = C.Logger("02b_extractor_nli")

MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
HIPOTESIS = {
    "ACL": "The anterior cruciate ligament is torn or injured.",
    "MCL": "The medial collateral ligament is torn or sprained.",
    "Medial Meniscus": "The medial meniscus is torn.",
    "Lateral Meniscus": "The lateral meniscus is torn.",
    "Medial OA": "There is osteoarthritis or cartilage loss in the medial compartment.",
    "Lateral OA": "There is osteoarthritis or cartilage loss in the lateral compartment.",
    "PF OA": "There is patellofemoral osteoarthritis or patellar cartilage damage.",
    "Effusion": "There is a joint effusion.",
    "Synovitis": "There is synovitis.",
    "Baker's": "There is a Baker's cyst.",
    "Contusion": "There is a bone bruise or bone marrow edema.",
    "Fracture": "There is a fracture.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--todos", action="store_true", help="procesar los 4.4k reportes (lento)")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    root = C.data_root()
    tr = pd.read_csv(root / "train.csv")
    if not args.todos:
        ids = set(tr.loc[tr[C.LABELS].notna().all(axis=1), C.STUDY_COL])
        sub = C.DIR_DATOS / "subconjunto.csv"
        if sub.exists():
            ids |= set(pd.read_csv(sub)[C.STUDY_COL])
        tr = tr[tr[C.STUDY_COL].isin(ids)]
    log(f"reportes a procesar: {len(tr)}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(dev).eval()
    if dev == "cuda":
        model.half()
    ent_idx = {v.lower(): k for k, v in model.config.id2label.items()}["entailment"]

    reps = tr[C.REPORT_COL].fillna("").astype(str).tolist()
    out = {C.STUDY_COL: tr[C.STUDY_COL].tolist()}
    for lab, hyp in HIPOTESIS.items():
        scores = []
        for i in range(0, len(reps), args.batch):
            enc = tok(reps[i:i + args.batch], [hyp] * len(reps[i:i + args.batch]), truncation="only_first",
                      max_length=512, padding=True, return_tensors="pt").to(dev)
            with torch.no_grad():
                logits = model(**enc).logits.float()
            # entailment vs contradiction (se ignora 'neutral' -> más robusto a reportes que no mencionan)
            con_idx = [k for k in range(logits.shape[1]) if k != ent_idx and
                       model.config.id2label[k].lower().startswith("contra")][0]
            p = torch.softmax(logits[:, [ent_idx, con_idx]], dim=1)[:, 0]
            scores.extend(p.cpu().numpy().tolist())
        out[f"score::{lab}"] = scores
        log(f"  {lab}: media={np.mean(scores):.3f}")
    pd.DataFrame(out).to_csv(C.DIR_DATOS / "etiquetas_debiles_nli.csv", index=False)
    C.append_bitacora(f"02b NLI ({MODEL}) sobre {len(tr)} reportes")


if __name__ == "__main__":
    main()
