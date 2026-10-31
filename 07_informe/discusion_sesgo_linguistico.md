# Entregable 7 — Riesgo de sesgo geográfico y lingüístico

> Texto de la sección 6 del informe (`Informe_P1_Rodilla_NeurIPS.pdf`). Cifras de
> `06_resultados/01_inventario.json` y `06_resultados/03_acuerdo_por_idioma.csv`.

Los estudios vienen de 16 centros en cinco continentes y cada reporte está en el idioma de su centro:
39 % inglés, 18 % español, 12 % turco, 7 % griego, 6 % alemán, 5 % ruso y el resto en otros idiomas.
Como el idioma coincide casi uno a uno con la institución, y esta con el fabricante del resonador y el
protocolo, idioma, sitio y equipo están confundidos. Cualquier diferencia de calidad del extractor
entre idiomas produce ruido de etiqueta correlacionado con el sitio, no ruido aleatorio.

Lo medimos en los dorados. El acuerdo del extractor es 0.81 en inglés (n = 28) y 0.80 en español
(n = 10), pero baja a 0.75 en turco, 0.67 en ruso, 0.61 en griego y 0.58 en los reportes sin idioma
identificado. La sensibilidad cae más: 0.78 en inglés, 0.66 en español y 0.64 en turco, frente a 0.42
en griego, 0.08 en ruso y 0.07 en "otro". Con 1–6 estudios por idioma fuera del inglés y el español
esto es un indicio, no una estimación, pero la dirección coincide con lo esperable. El léxico es más
completo en inglés y español; en ruso, griego y turco la morfología es más rica y la negación suele
ir después del término, y además el formato del reporte cambia por centro: unos listan negativos
explícitos y otros solo escriben lo patológico en dos o tres líneas.

**Consecuencia para el modelo.** Si en un centro los positivos quedan sub-etiquetados, la red aprende
que las imágenes de ese escáner tienden a ser normales y puede usar atajos (textura, contraste, campo
de visión) que predicen el sitio en vez de la anatomía, como documentaron Zech et al. (2018). El
AUC agregado puede verse bien mientras el desempeño en el centro peor etiquetado es malo; los
mapas de A1 sobre bordes y fondo muestran que este tipo de atajo es posible.

**Implicancias para un despliegue real.** Un hospital peruano, con reportes en español y otro equipo
y protocolo, sería en la práctica un sitio nuevo. Los falsos negativos se concentrarían en hallazgos
que los radiólogos suelen omitir en el texto (sinovitis, artrosis leve), justo los peor representados
en el entrenamiento. Antes de usarlo haría falta: validación local con etiquetas de imagen hechas
por radiólogos del centro; métricas por sitio, idioma y fabricante, no solo el promedio; recalibrar
los umbrales; y monitorear el desempeño. En el entrenamiento ayudaría validar agrupando por sitio,
revisar el extractor de cada idioma con hablantes nativos o traducir los reportes a un idioma pivote
con un modelo local.
