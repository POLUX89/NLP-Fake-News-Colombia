# Data Statement — ColombiaCheck Verdict Corpus

> Plantilla basada en *Data Statements for NLP* (Bender & Friedman, 2018).
> Los campos `TODO` se completan tras la Fase 2.

## A. Curation Rationale
Afirmaciones seleccionadas y chequeadas por ColombiaCheck, típicamente
declaraciones de figuras públicas colombianas sobre política y actualidad. El
criterio de inclusión es **el de ColombiaCheck** (qué deciden chequear), no un
muestreo representativo del discurso público → sesgo de selección inherente.

## B. Language Variety
- Español (`es`), variedad **colombiana** (es-CO).
- Registro: periodístico / declaraciones públicas parafraseadas.

## C. Speaker / Author Demographics
- Los textos de las afirmaciones son redactados por el equipo editorial de
  ColombiaCheck (no transcripciones literales del hablante original).
- Demografía de autores/editoras: `TODO` (no publicada; no inferir).

## D. Annotator Demographics
- Las etiquetas (veredictos) las asignan verificadores profesionales de
  ColombiaCheck según su metodología pública. Demografía individual: `TODO` /
  no disponible.

## E. Speech Situation
- Contexto: verificación periodística de declaraciones públicas.
- Ventana temporal: ~`2020-03-24` → `2026-02-27` (rango del muestreo; n=4.756).
- Asincrónico, escrito, editado.

## F. Text Characteristics
- Frases cortas a medias (una afirmación por instancia).
- Vocabulario político/institucional colombiano; nombres propios frecuentes.

## G. Recording Quality / Provenance
- Origen: marcado **ClaimReview** (schema.org) en páginas públicas y/o Google
  Fact Check Tools API. Sin audio; texto ya digital.

## H. Annotation Guidelines
- Metodología de ColombiaCheck (escala de veredictos). Este proyecto **no**
  reetiqueta; usa las etiquetas tal como se publican.

## I. Known Limitations / Bias
- **Desbalance** fuerte hacia `Falso` (65 %) frente a `Verdadero` (3.4 %).
- Sesgo de selección (qué se elige chequear) y de una **única** organización.
- El modelo entrenado sobre este corpus reproduce ese juicio; **no** mide verdad.
