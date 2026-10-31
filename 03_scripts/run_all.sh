#!/usr/bin/env bash
# Ejecuta TODO el pipeline P1 en orden (Kaggle Notebook con GPU T4, Internet ON).
# Tiempo estimado total: ~60-90 min (cache ~10 min, 6 runs x ~5-8 min, NLI ~10 min).
# Si algo falla, cada paso se puede correr suelto; los logs quedan en 02_logs/.
set -euo pipefail
cd "$(dirname "$0")"

python 01_inventario_metadatos.py
python 02_extractor_etiquetas_debiles.py
python 04_seleccion_subconjunto.py
# opcional (segundo método de extracción, ~10 min en GPU; requiere Internet ON):
python 02b_extractor_nli_multilingue.py || echo "[aviso] NLI omitido"
python 03_validar_extractor_vs_dorados.py

python 05_cache_dicom_25d.py --slot sag_fluid
python 05_cache_dicom_25d.py --slot cor_fluid

# --- Entregable 2: dos arquitecturas ---
python 06_entrenar_evaluar.py --run A1_scratch  --arch scratch  --epochs 15
python 06_entrenar_evaluar.py --run A2_resnet50 --arch resnet50
# --- Entregable 4: ablaciones (factor único vs A2) ---
python 06_entrenar_evaluar.py --run B1_focal    --arch resnet50 --loss focal
python 06_entrenar_evaluar.py --run B2_altaconf --arch resnet50 --ruido alta_conf
python 06_entrenar_evaluar.py --run B3_coronal  --arch resnet50 --slot cor_fluid
# --- Entregable 6: mitigación del desbalance (A2 usa pos_weight; aquí sin él y con sobremuestreo) ---
python 06_entrenar_evaluar.py --run C1_sin_posw --arch resnet50 --pos_weight none
python 06_entrenar_evaluar.py --run C2_sampler  --arch resnet50 --sampler raros
# --- opcional: profundidad del backbone ---
# python 06_entrenar_evaluar.py --run D1_resnet18 --arch resnet18

# --- Entregable 5: Grad-CAM ---
python 07_gradcam.py --run A2_resnet50
python 07_gradcam.py --run A1_scratch

# --- Tablas y figuras para el informe ---
python 08_tablas_para_informe.py --base A2_resnet50
echo "LISTO. Resultados en 06_resultados/, logs en 02_logs/"
