# Taller XAI sin dependencias externas

Este proyecto implementa un flujo completo de modelado supervisado, explicabilidad y reflexión ética usando **únicamente la biblioteca estándar de Python**. Se entrena una regresión logística propia sobre el dataset `cybersecurity synthesized data.csv` (subido al repositorio) y se generan visualizaciones SVG sin dependencias adicionales.

## Estructura
- `analysis/main.py`: carga el dataset de ciberseguridad, entrena el modelo, calcula métricas y crea las visualizaciones (coeficientes, Permutation Feature Importance y PDP).
- `data/cybersecurity synthesized data.csv`: 500 eventos de red sintéticos etiquetados con probables intrusiones.
- `outputs/`: contiene las gráficas SVG exportadas.
- `PRUEBA`: marcador del repositorio original.

## Metodología
1. **Calidad y sesgos de datos**
   - Dataset sintético de tráfico y actividad de red con duración, bytes, fallos de autenticación, alertas, destinos internos y accesos a recursos sensibles.
   - Estandarización de columnas y división estratificada simple para evitar fugas de información.
   - Modelo regularizado (L2) para mitigar sobreajuste y pesos extremos.
2. **Entrenamiento**
   - Regresión logística implementada con descenso de gradiente y cálculo manual de la función de pérdida logística.
   - División 80/20 entrenamiento-prueba con semilla fija.
3. **Explicabilidad aplicada**
   - **Interpretación de coeficientes**: magnitud absoluta de los pesos del modelo.
   - **Permutation Feature Importance**: caída de exactitud al permutar cada columna.
   - **Partial Dependence Plots (PDP)** para bytes salientes y conteo de alertas.
   - **Explicaciones individuales** vía contribuciones `peso * valor` y probabilidad estimada.
4. **Evaluación**
   - Métricas exactitud, precisión, recall y F1 calculadas manualmente.
   - Ejecución de ejemplo produce una exactitud aproximada de 0.93 en el conjunto de prueba.

## Cómo ejecutar
```bash
python analysis/main.py
```
El script leerá `data/cybersecurity synthesized data.csv`, imprimirá métricas y escribirá las imágenes SVG en `outputs/`.

## Reflexión ética y social
- **Transparencia**: al implementar el algoritmo desde cero y exponer coeficientes, se facilita la trazabilidad de cada decisión.
- **Riesgos**: un sesgo hacia alertas y fallos de autenticación podría generar demasiados falsos positivos, afectando la operación normal o usuarios legítimos.
- **Mitigaciones**: limitar pesos mediante regularización, monitorear importancia por permutación para detectar dependencia excesiva de una sola variable y documentar las decisiones.
- **Sin explicabilidad**: desplegar el modelo como caja negra dificultaría justificar bloqueos de tráfico o acceso, y podría esconder dependencias peligrosas (p. ej. sobre un único tipo de alerta).

## Resultados destacados
- Mayor peso en **fallos de autenticación** y **cantidad de alertas**, moderado en **tráfico de salida**.
- PDP muestra que más bytes salientes o más alertas incrementan la probabilidad estimada de intrusión.
- Ejemplos individuales evidencian cómo cada característica contribuye a la probabilidad final, útil para auditorías.
