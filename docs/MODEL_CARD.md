# Model Card — ColombiaCheck Verdict Classifier

> Plantilla basada en *Model Cards for Model Reporting* (Mitchell et al., 2019).
> Los campos marcados con `TODO` se completan al terminar la Fase 3.

## Model Details
- **Desarrollado por:** Daniel Felipe Sacristán Ávila (portfolio personal).
- **Fecha:** `TODO`
- **Versión:** 0.1.0 (sin entrenar aún).
- **Tipo:** Clasificador de texto multiclase; fine-tuning de **BETO**
  (`dccuchile/bert-base-spanish-wwm-cased`).
- **Entrada:** texto de la afirmación chequeada (`claim_reviewed`), español.
- **Salida:** veredicto de ColombiaCheck ∈ {`Falso`, `Cuestionable`,
  `Verdadero`, `Verdadero pero`, `Chequeo Múltiple`}.
- **Licencia:** MIT (código). Pesos: `TODO`.

## Intended Use
- **Uso previsto:** demostración educativa / de portfolio de un pipeline de NLP
  en español, y análisis de cómo un modelo reproduce las etiquetas de una
  organización de fact-checking.
- **Usuarios previstos:** reclutadores técnicos, estudiantes de NLP, el autor.
- **Fuera de alcance (NO usar para):**
  - Determinar si una afirmación es objetivamente verdadera o falsa.
  - Moderación de contenido, decisiones editoriales o periodísticas automáticas.
  - Cualquier decisión sobre personas, medios o publicaciones.

## Factors
- Dominio: política y actualidad colombiana; declaraciones de figuras públicas.
- El modelo puede depender del **tema** y del **lenguaje** más que del valor de
  verdad. `TODO`: análisis por subgrupo (tema, periodo, longitud del claim).

## Metrics
- `TODO`: macro-F1 (métrica principal por el desbalance), F1 por clase,
  matriz de confusión, accuracy balanceada.
- Baseline de comparación: `TODO` (p. ej. TF-IDF + regresión logística).

## Evaluation & Training Data
- Fuente: chequeos de ColombiaCheck (ver [Datasheet](DATASHEET.md)).
  **4.756** chequeos; ~68 % con `claim_reviewed` estructurado (≈3.200 usables).
- Split: `TODO` (estratificado por veredicto; **separación temporal** para
  evitar fuga de tema entre train/test).
- Desbalance: `Falso` 65 % / `Cuestionable` 19.8 % / `Chequeo Múltiple` 5.7 % /
  `Verdadero pero` 4.1 % / `Verdadero` 3.4 % → estrategia (class weights /
  re-muestreo / colapso de clases minoritarias) a decidir en Fase 3.

## Ethical Considerations
- El modelo **no** verifica hechos; reproduce el juicio de una sola organización
  y hereda sus sesgos de selección (qué se elige chequear) y de etiquetado.
- Riesgo de mal uso: presentarlo como "detector de verdad". Mitigación: esta
  Model Card, el README y el [Data Statement](DATA_STATEMENT.md).

## Caveats & Recommendations
- Corpus de tamaño moderado (~4.756) pero sesgado y de una sola fuente →
  los resultados no generalizan fuera de ColombiaCheck.
- Recomendado como artefacto didáctico, no productivo.
