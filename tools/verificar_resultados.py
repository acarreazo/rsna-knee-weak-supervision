"""Verificación de resultados — no necesita GPU, datos de la competencia ni PyTorch.

Dos usos:

  1) Coherencia interna de los resultados versionados (≈ 1 s, solo numpy + pandas):
         python tools/verificar_resultados.py

     Recalcula los agregados a partir de los archivos por run (02_logs/runs/<run>/metricas.json y
     metricas_por_hallazgo.csv) y comprueba que coinciden con las tablas consolidadas de
     06_resultados/ (que son las que cita el informe).

  2) Comparar una ejecución nueva contra la ejecución oficial (30-oct-2026, Kaggle T4):
         python tools/verificar_resultados.py --nuevo /kaggle/working/p1

     * Pasos deterministas (inventario, extractor de reglas, selección del subconjunto): deben
       coincidir exactamente.
     * Entrenamiento en GPU: cuDNN no es determinista, así que no se espera igualdad bit a bit.
       Se considera reproducido si el AUC macro dorado nuevo cae dentro del IC 95 % bootstrap
       de la ejecución oficial.

Código de salida: 0 si todo pasa, 1 si alguna verificación falla.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

fallas: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(("  [OK]    " if ok else "  [FALLA] ") + msg)
    if not ok:
        fallas.append(msg)


def cargar_runs(base: Path) -> dict[str, dict]:
    return {p.parent.name: json.load(open(p, encoding="utf-8"))
            for p in sorted((base / "02_logs" / "runs").glob("*/metricas.json"))}


def coherencia_interna(base: Path) -> dict[str, dict]:
    res = base / "06_resultados"
    runs = cargar_runs(base)
    print(f"\n== Coherencia interna de {base} ({len(runs)} runs) ==")
    check(len(runs) == 7, f"7 runs esperados (A1, A2, B1–B3, C1, C2); encontrados: {list(runs)}")

    comp = pd.read_csv(res / "08_comparacion_runs.csv").set_index("run")
    auc_tab = pd.read_csv(res / "08_auc_por_hallazgo.csv", index_col=0)
    for r, m in runs.items():
        por_h = pd.read_csv(base / "02_logs" / "runs" / r / "metricas_por_hallazgo.csv").set_index("hallazgo")
        macro_csv = por_h.loc[LABELS, "auc"].mean()
        check(abs(macro_csv - m["auc_macro_dorado"]) < 1e-6,
              f"{r}: AUC macro = media de las 12 AUC por hallazgo ({m['auc_macro_dorado']:.4f})")
        check(abs(comp.loc[r, "auc_macro_dorado"] - round(m["auc_macro_dorado"], 4)) < 1e-4,
              f"{r}: 08_comparacion_runs.csv coincide con metricas.json")
        dif = max(abs(auc_tab.loc[l, r] - round(m["auc_por_hallazgo"][l], 3)) for l in LABELS)
        check(dif < 1.5e-3, f"{r}: 08_auc_por_hallazgo.csv coincide hallazgo por hallazgo")
        lo, hi = m["ic95_dorado"]
        check(lo <= m["auc_macro_dorado"] <= hi, f"{r}: el AUC puntual está dentro de su IC 95 % [{lo:.3f}, {hi:.3f}]")

    ext = pd.read_csv(res / "03_validacion_extractor.csv")
    resumen = json.load(open(res / "03_resumen.json", encoding="utf-8"))
    for metodo in ext["metodo"].unique():
        a = ext.loc[ext.metodo == metodo, "auc"].mean()
        check(abs(a - resumen[metodo]["auc_macro"]) < 1e-6,
              f"extractor '{metodo}': AUC macro = media por hallazgo ({a:.3f})")
    return runs


def tabla_resumen(runs: dict[str, dict], base: Path) -> None:
    filas = [{"run": r, "arquitectura": m["arch"], "plano": m["slot"], "pérdida": m["loss"],
              "ruido": m["ruido"], "pos_weight": m["pos_weight"], "sampler": m["sampler"],
              "AUC dorado": round(m["auc_macro_dorado"], 3),
              "IC 95 %": f"[{m['ic95_dorado'][0]:.3f}, {m['ic95_dorado'][1]:.3f}]",
              "F1 macro": round(m["f1_macro_dorado"], 3), "n dorado": m["n_dorado"],
              "min": round(m["min_entrenamiento"], 1)} for r, m in runs.items()]
    print("\n== Resumen por run (conjunto dorado) ==")
    print(pd.DataFrame(filas).to_string(index=False))
    abl = base / "06_resultados" / "08_ablaciones_pareadas.csv"
    if abl.exists():
        print("\n== Δ AUC pareado vs A2_resnet50 (bootstrap sobre los mismos estudios dorados) ==")
        print(pd.read_csv(abl).to_string(index=False))


def comparar(oficial: Path, nuevo: Path) -> None:
    print(f"\n== Ejecución nueva ({nuevo}) vs oficial ==")
    # deterministas
    for f in ["04_prevalencia.csv"]:
        a, b = oficial / "06_resultados" / f, nuevo / "06_resultados" / f
        if b.exists():
            check(pd.read_csv(a).equals(pd.read_csv(b)), f"{f}: idéntico (selección del subconjunto determinista)")
        else:
            check(False, f"{f}: no existe en la ejecución nueva")
    a, b = oficial / "06_resultados" / "03_resumen.json", nuevo / "06_resultados" / "03_resumen.json"
    if b.exists():
        ra, rb = json.load(open(a)), json.load(open(b))
        check(abs(ra["reglas"]["auc_macro"] - rb["reglas"]["auc_macro"]) < 1e-9,
              f"extractor de reglas: AUC idéntico ({rb['reglas']['auc_macro']:.4f})")
    # entrenamiento
    ro, rn = cargar_runs(oficial), cargar_runs(nuevo)
    filas = []
    for r, m in ro.items():
        if r not in rn:
            check(False, f"{r}: falta en la ejecución nueva")
            continue
        lo, hi = m["ic95_dorado"]
        nv = rn[r]["auc_macro_dorado"]
        filas.append({"run": r, "AUC oficial": round(m["auc_macro_dorado"], 4), "IC 95 % oficial": f"[{lo:.3f}, {hi:.3f}]",
                      "AUC nuevo": round(nv, 4), "Δ": round(nv - m["auc_macro_dorado"], 4)})
        check(lo <= nv <= hi, f"{r}: AUC nuevo {nv:.4f} dentro del IC 95 % oficial")
    if filas:
        print(pd.DataFrame(filas).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nuevo", type=Path, help="carpeta de trabajo de una ejecución nueva (p. ej. /kaggle/working/p1)")
    ap.add_argument("--oficial", type=Path, default=REPO, help="carpeta con 02_logs/ y 06_resultados/ oficiales")
    args = ap.parse_args()

    runs = coherencia_interna(args.oficial)
    tabla_resumen(runs, args.oficial)
    if args.nuevo:
        comparar(args.oficial, args.nuevo)

    print(f"\n{'TODO OK' if not fallas else f'{len(fallas)} verificación(es) fallaron'}")
    sys.exit(1 if fallas else 0)


if __name__ == "__main__":
    main()
