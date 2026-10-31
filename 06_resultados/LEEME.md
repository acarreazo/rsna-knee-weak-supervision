# 06_resultados — tablas y figuras de la ejecución oficial

Prefijo = paso que las genera.

| Archivo | Entregable |
|---|---|
| `01_inventario.json` | Inventario: estudios, dorados, idiomas, planos |
| `02_extractor_resumen.json` | Prevalencia débil y tasa de mención por hallazgo |
| `03_validacion_extractor.csv`, `03_resumen.json`, `03_acuerdo_por_idioma.csv`, `figuras/03_*.png` | 1 y 7 |
| `04_prevalencia.csv` | 6 |
| `07_gradcam_<run>.csv`, `figuras/07_*.png` | 5 |
| `08_comparacion_runs.(csv\|md\|tex)`, `08_auc_por_hallazgo.(csv\|tex)` | 2 y 3 |
| `08_ablaciones_pareadas.csv` | 4 |
| `08_desbalance.csv` | 6 |
| `figuras/08_*.png` | 2, 3, 4 |

En `07_gradcam_<run>.csv` se eliminó la columna `StudyInstanceUID` antes de publicar (D010).
