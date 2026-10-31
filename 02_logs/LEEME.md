# 02_logs — trazabilidad de la ejecución oficial (Kaggle, GPU T4, 30-oct-2026)

- `bitacora_ejecuciones.md`: una línea por paso/run, escrita automáticamente por los scripts.
- `<NN>_<script>.log`: salida completa de cada script.
- `runs/<run>/`: `config.json` (argumentos exactos), `train.log`, `historial.csv` (AUC val por época),
  `metricas_por_hallazgo.csv` (AUC, umbral, P/R/F1 por hallazgo) y `metricas.json` (resumen del run).

**No se versionan** (ver `01_decisiones/decisiones.md`, D010): `mejor.pt` (pesos) y
`pred_val.csv` / `pred_dorado.csv`, porque contienen `StudyInstanceUID` y las etiquetas doradas por
estudio, que son datos de la competencia. Se regeneran al ejecutar el pipeline.

Las **decisiones** (por qué se hizo algo) van en `01_decisiones/decisiones.md`, no aquí.
