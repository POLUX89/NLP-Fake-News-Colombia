# Datasheet — ColombiaCheck Verdict Corpus

> Plantilla basada en *Datasheets for Datasets* (Gebru et al., 2021).
> Los campos `TODO` se completan tras la Fase 2 (construcción del corpus).

## Motivation
- **¿Para qué se creó?** Entrenar/evaluar un clasificador NLP del veredicto de
  chequeos y servir de artefacto de portfolio con documentación de gobernanza.
- **¿Quién lo creó?** Daniel Felipe Sacristán Ávila.
- **¿Quién lo financió?** Proyecto personal, sin financiación.

## Composition
- **¿Qué representa cada instancia?** Una afirmación chequeada por ColombiaCheck
  con su veredicto.
- **Campos:** `url`, `claim_reviewed` (texto), `verdict` (etiqueta),
  `pub_date`, `jsonld_types`. (`claimant`, `tags` descartados: no fiables.)
- **Número de instancias:** **4.756** chequeos únicos (recon `2026-07-20`).
- **Etiquetas y distribución:** `Falso` 65.0 % (3.093), `Cuestionable` 19.8 %
  (940), `Chequeo Múltiple` 5.7 % (273), `Verdadero pero` 4.1 % (197),
  `Verdadero` 3.4 % (161), sin etiqueta 1.9 % (92). Desbalance fuerte.
- **¿Datos faltantes?** `claim_reviewed` y `pub_date` ~68 % de cobertura (donde
  hay ClaimReview; menor en artículos antiguos). `claimant`/`tags`: 0 %.
- **¿Información sensible/personal?** Contiene nombres de figuras públicas en el
  texto de las afirmaciones. No hay datos personales privados.

## Collection Process
- **¿Cómo se obtuvo?** Extracción de **marcado ClaimReview (schema.org JSON-LD)**
  en las páginas públicas de chequeos y/o Google Fact Check Tools API.
  Preferencia por datos estructurados frente a scraping de texto completo.
- **Herramienta:** [`colombiacheck_recon.py`](../colombiacheck_recon.py).
- **Politeness:** respeta `robots.txt` (`/chequeos` permitido), `User-Agent`
  identificado, caché local, throttling 1.5 s.
- **Ventana temporal:** ~`2020-03-24` → `2026-02-27` (rango del muestreo de
  50 artículos; el rango completo puede ser mayor).

## Preprocessing / Cleaning / Labeling
- `TODO`: normalización de etiquetas legacy (pre-2018), tratamiento de
  `Chequeo Múltiple` (formato, no veredicto simple), deduplicación por `url`.

## Uses
- Clasificación de veredicto; análisis de sesgo de selección.
- **Usos no recomendados:** afirmar verdad objetiva; generalizar a otros medios.

## Distribution & Maintenance
- **¿Se redistribuye el corpus?** **No** se redistribuye el texto de los chequeos
  (propiedad de ColombiaCheck). El repo incluye el **código** para reconstruirlo.
- **Contacto de la fuente:** `contacto@colombiacheck.com` (a contactar antes de
  cualquier cosecha de texto completo).
- **Mantenedor:** el autor.
