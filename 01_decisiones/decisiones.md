# Registro de decisiones — P1 RSNA Knee (formato ADR corto)

Cada decisión: **contexto → decisión → alternativas descartadas → evidencia → estado**.
Si una decisión cambia, NO se borra: se marca *Reemplazada por Dxxx* y se agrega la nueva.
Fecha de apertura del registro: 2026-10-29.

---

## D001 — Dónde se procesa: Kaggle Notebooks, no descarga local
- **Contexto:** 569.76 GB / 819,640 DICOM. Entrega 30-oct. Ninguna PC del equipo tiene GPU ni espacio.
- **Decisión:** todo corre en un Kaggle Notebook (GPU **T4**, Internet ON) con la competencia montada en `/kaggle/input`. Solo bajamos tablas, figuras y logs (zip de ~MB).
- **Descartado:** descargar un subconjunto con la API de Kaggle (horas de descarga + necesidad de GPU propia); P100 (PyTorch reciente ya no la soporta).
- **Riesgo abierto:** la competencia cerró el 22-oct. **Verificar hoy** que la pestaña *Data* sigue accesible para la cuenta de quien corre el notebook (aceptar *Rules*). Si no, avisar al profesor de inmediato.
- **Estado:** vigente.

## D002 — Esquema de etiquetas y conjunto dorado
- **Contexto:** el enunciado pide confirmar el esquema en la pestaña *Data*.
- **Decisión:** `train.csv` contiene `StudyInstanceUID`, `Report` y 12 columnas: `ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture`. Estudio **dorado** = fila con las 12 etiquetas no nulas (~58). `train_series.csv` trae `Anatomical_Plane`, `Fluid_Sensitive`, `Fat_Suppression` por serie.
- **Evidencia:** código y documentación pública de participantes ([JunhaoLiXD](https://github.com/JunhaoLiXD/RSNA_Knee_Abnormality_Detection), [homeshwarnelakurthi](https://github.com/homeshwarnelakurthi/RSNA-Knee-Abnormality-Detection)). `01_inventario_metadatos.py` aborta si alguna columna no existe → la confirmación queda en `02_logs/01_inventario_metadatos.log`.
- **Hechos publicados por otros equipos que condicionan el diseño** (citar en el informe): las etiquetas doradas son *derivadas de la imagen* por 2 radiólogos MSK + adjudicador con umbrales de severidad; el acuerdo reporte→dorado ronda ~82 %; los 58 dorados están ~2× enriquecidos en positivos; los reportes sub-reportan hallazgos como sinovitis.
- **Estado:** vigente (confirmar con el log del paso 01).

## D003 — Extractor de etiquetas débiles
- **Decisión:** método principal = **reglas multilingües** (en, de, es, pt, fr, nl, it, tr, el, ru) con negación tipo NegEx, duda y severidad; salida **ordinal** (0.05 negado … 0.95 severo) + **confianza**. Método secundario opcional = **NLI zero-shot multilingüe** (mDeBERTa-v3 XNLI, pesos abiertos, corre dentro del notebook) y **ensamble** por promedio de rangos.
- **Por qué ordinal:** el AUC solo usa el orden; una salida binaria empata a todos los positivos.
- **"No mencionado":** no es un negativo seguro en reportes cortos → confianza 0.6 si el reporte tiene ≥80 palabras, 0.3 si no. Esa confianza pondera la pérdida (D008).
- **Descartado:** enviar los reportes a un LLM por API (las reglas de la competencia lo prohíben y es costoso).
- **Validación:** `03_validar_extractor_vs_dorados.py` → AUC, acuerdo, sensibilidad, especificidad, kappa y **tasa de ruido = 1 − acuerdo** por hallazgo; acuerdo por idioma.
- **Estado:** vigente.

## D004 — Reducción de escala y particiones
- **Decisión:** pool = estudios **no dorados** con reporte y con series sagital **y** coronal sensibles a líquido. Se toman **500** estudios: primero hasta 25 positivos débiles de cada hallazgo (del más raro al más común), luego relleno aleatorio. Partición **train 85 % / val 15 %** estratificada por el hallazgo positivo más raro. Los **58 dorados = test limpio**, nunca se usan para elegir hiperparámetros, umbrales ni épocas.
- **Descartado:** usar los dorados para early stopping (n=58, sobreajuste garantizado).
- **Limitación conocida:** no agrupamos por sitio/escáner (otro equipo midió +0.05 de AUC "fantasma" con K-fold aleatorio). Lo declaramos en el informe como amenaza a la validez.
- **Estado:** vigente.

## D005 — Plano, secuencia y preprocesamiento 2.5D
- **Decisión:** slot principal **sagital PD/T2 con supresión grasa** (`Anatomical_Plane=Sagittal`, `Fluid_Sensitive=1`); ablación con **coronal sensible a líquido**. Por estudio: 9 cortes uniformes del 80 % central, recorte físico central de 150 mm, caché 256 px, entrenamiento a 224 px.
- **Lateralidad:** tag DICOM `Laterality` si existe; si no, signo de la x del centro de imagen (LPS: +x = izquierda). Se canonicaliza: sagital ordenado **medial→lateral**; coronal espejado en rodillas derechas (medial siempre a la izquierda).
- **Prohibido:** flip horizontal como aumento (intercambia medial/lateral → corrompe 5 de 12 etiquetas).
- **Estado:** vigente.

## D006 — Arquitecturas
- **Decisión:** marco común "encoder 2D por corte + atención por hallazgo" (MIL 2.5D): cada corte entra con sus vecinos como 3 canales; 12 cabezas de atención eligen qué cortes mirar; 12 salidas sigmoides.
  - **A1 — CNN desde cero:** 5 bloques (conv-BN-ReLU)×2 + maxpool, ~1.9 M parámetros.
  - **A2 — ResNet-50 ImageNet** (fine-tuning completo, LR del backbone = LR cabeza / 10).
  - Opcionales: ResNet-18 (profundidad), EfficientNet-B0.
- **Por qué atención:** da interpretabilidad a dos niveles (qué corte + Grad-CAM dentro del corte).
- **Estado:** vigente.

## D007 — Evaluación y umbral
- **Decisión:** métrica oficial = **AUC-ROC macro** (12 hallazgos, peso igual) sobre los dorados, con desglose por hallazgo e **IC 95 % bootstrap** (1000 remuestreos). P/R/F1 con **umbral por hallazgo que maximiza F1 en val débil** (elegido fuera del test; también se reporta F1@0.5).
- **Advertencia para el informe:** con 58 casos y hallazgos de 5–15 positivos, un AUC por hallazgo tiene IC de ±0.1–0.2 → interpretar direcciones, no décimas.
- **Estado:** vigente.

## D008 — Ablaciones (un factor a la vez contra A2)
| Run | Factor | Variante |
|---|---|---|
| B1_focal | pérdida | focal (γ=2) vs BCE |
| B2_altaconf | etiquetas ruidosas | solo etiquetas con confianza ≥0.6 vs ponderar por confianza |
| B3_coronal | plano | coronal FS vs sagital FS |
| D1_resnet18 (opc.) | profundidad | ResNet-18 vs ResNet-50 |
- Comparación con **Δ AUC y IC bootstrap pareado** sobre los mismos estudios dorados (`08_ablaciones_pareadas.csv`).
- **Estado:** vigente.

## D009 — Desbalance de clases
- **Decisión:** mitigación por defecto = `pos_weight = sqrt(neg/pos)` acotado a [1, 10] en BCE/focal. Se evalúa contra **C1 (sin pos_weight)** y **C2 (sobremuestreo ×4 de estudios con hallazgos del tercio más raro)**, mirando el AUC de los hallazgos raros (fractura, Baker, LCM, artrosis lateral).
- **Estado:** vigente.

## D010 — Datos sensibles y repositorio
- **Decisión:** el repo de GitHub incluye código, notebook, decisiones, logs de texto, tablas agregadas y figuras. **No** se suben DICOM, `05_datos/*.csv` (contienen UIDs y texto derivado de reportes), cachés ni pesos `.pt` (`.gitignore`). Revisar *Rules* de la competencia sobre uso de datos fuera de ella.
- **Estado:** vigente.

## D011 — Priorización por plazo (entrega 30-oct)
- **Decisión:** orden de ejecución = 01→02→04→03→05 → **A1, A2** (mínimo entregable) → B1, B2 → B3, C1, C2 → Grad-CAM → tablas. Si el tiempo de GPU no alcanza, recortar en ese orden inverso y declararlo.
- **Estado:** vigente.
