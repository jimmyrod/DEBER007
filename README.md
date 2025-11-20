# Taller XAI sin dependencias externas

Este proyecto implementa un flujo completo de modelado supervisado, explicabilidad y reflexión ética usando **únicamente la biblioteca estándar de Python**. Se entrena una regresión logística propia sobre un conjunto de crédito sintético y se generan visualizaciones SVG sin dependencias adicionales.

## Estructura
- `analysis/main.py`: genera el dataset, entrena el modelo, calcula métricas y crea las visualizaciones (coeficientes, Permutation Feature Importance y PDP).
- `data/synthetic_credit.csv`: muestra los 600 registros sintéticos generados.
- `outputs/`: contiene las gráficas SVG exportadas.
- `PRUEBA`: marcador del repositorio original.

## Metodología
1. **Calidad y sesgos de datos**
   - Datos sintéticos con variables de edad, ingreso, razón de deuda, historial de impagos, años trabajados y educación.
   - Estandarización de columnas y división estratificada simple para evitar fugas de información.
   - Modelo regularizado (L2) para mitigar sobreajuste y pesos extremos.
2. **Entrenamiento**
   - Regresión logística implementada con descenso de gradiente y cálculo manual de la función de pérdida logística.
   - División 80/20 entrenamiento-prueba con semilla fija.
3. **Explicabilidad aplicada**
   - **Interpretación de coeficientes**: magnitud absoluta de los pesos del modelo.
   - **Permutation Feature Importance**: caída de exactitud al permutar cada columna.
   - **Partial Dependence Plots (PDP)** para ingreso y razón de deuda.
   - **Explicaciones individuales** vía contribuciones `peso * valor` y probabilidad estimada.
4. **Evaluación**
   - Métricas exactitud, precisión, recall y F1 calculadas manualmente.
   - Ejecución de ejemplo produce una exactitud aproximada de 0.93 en el conjunto de prueba.

## Cómo ejecutar
```bash
python analysis/main.py
```
El script generará/actualizará `data/synthetic_credit.csv`, imprimirá métricas y escribirá las imágenes SVG en `outputs/`.

## Reflexión ética y social
- **Transparencia**: al implementar el algoritmo desde cero y exponer coeficientes, se facilita la trazabilidad de cada decisión.
- **Riesgos**: variables como ingreso o historial de impagos podrían amplificar desigualdades socioeconómicas. Es crucial validar que el dataset sintético o real no codifique discriminación indirecta.
- **Mitigaciones**: limitar pesos mediante regularización, monitorear importancia por permutación para detectar dependencia excesiva de una sola variable y documentar las decisiones.
- **Sin explicabilidad**: desplegar el modelo como caja negra dificultaría justificar rechazos crediticios y podría incumplir principios de equidad y regulaciones de transparencia.

## Resultados destacados
- Mayor peso positivo en **pocos impagos previos** y **mayor ingreso**, negativo en **razón de deuda** alta.
- PDP muestra que aumentar el ingreso estandarizado sube la probabilidad de buen crédito y que la razón de deuda alta la reduce.
- Ejemplos individuales evidencian cómo cada característica contribuye a la probabilidad final, útil para auditorías.
