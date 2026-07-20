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
- **Número de instancias:** `TODO` (recon en curso; ~9 fichas/página, sin fin
  de archivo hasta la última página crawleada).
- **Etiquetas y distribución:** `TODO`. Predominio observado de `Falso`.
- **¿Datos faltantes?** `pub_date` ~80–90 % de cobertura; `claim_reviewed`
  ~80 % (donde hay ClaimReview).
- **¿Información sensible/personal?** Contiene nombres de figuras públicas en el
  texto de las afirmaciones. No hay datos personales privados.

## Collection Process
- **¿Cómo se obtuvo?** Extracción de **marcado ClaimReview (schema.org JSON-LD)**
  en las páginas públicas de chequeos y/o Google Fact Check Tools API.
  Preferencia por datos estructurados frente a scraping de texto completo.
- **Herramienta:** [`colombiacheck_recon.py`](../colombiacheck_recon.py).
- **Politeness:** respeta `robots.txt` (`/chequeos` permitido), `User-Agent`
  identificado, caché local, throttling 1.5 s.
- **Ventana temporal:** `TODO` (rango de `pub_date`).

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
