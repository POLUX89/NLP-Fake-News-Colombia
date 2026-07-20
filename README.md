# Fake News Colombia — NLP verdict classification over ColombiaCheck

Clasificación NLP del **veredicto** de chequeos de [ColombiaCheck](https://colombiacheck.com)
(`Falso` / `Cuestionable` / `Verdadero` / `Verdadero pero` / `Chequeo Múltiple`),
acompañada de documentación de gobernanza de datos y modelos: **Model Card**,
**Datasheet** y **Data Statement**.

> ⚠️ **Qué es y qué no es este proyecto.**
> El modelo aprende a reproducir **cómo ColombiaCheck etiquetó** cada afirmación,
> a partir del texto de la afirmación. **No** es un detector de verdad ni un
> "detector de noticias falsas": aprende rasgos superficiales y temas
> correlacionados con las etiquetas de una única organización. Esta limitación se
> documenta explícitamente en la [Model Card](docs/MODEL_CARD.md) y el
> [Data Statement](docs/DATA_STATEMENT.md). Tratarla con honestidad es parte
> central del objetivo de este repositorio.

## Fases del proyecto

| Fase | Estado | Entregable |
|------|--------|-----------|
| 0 · Entorno + repo | ✅ | `pyproject.toml` (Python 3.13), estructura, `.gitignore` |
| 1 · Reconocimiento | ✅ | [`colombiacheck_recon.py`](colombiacheck_recon.py) → `recon_output/` |
| 2 · Adquisición de datos | ⬜ | Corpus `(afirmación → veredicto)` desde datos estructurados / Fact Check Tools API |
| 3 · Modelo NLP | ⬜ | Fine-tuning de BETO (BERT en español), clasificación multiclase |
| 4 · Gobernanza + demo | ⬜ | Model Card · Datasheet · Data Statement · app Streamlit |

## Hallazgos del reconocimiento (Fase 1)

<!-- NUMEROS_RECON: se rellenan al terminar el recon completo -->
- **Marcado ClaimReview** presente en ~80 % de los artículos muestreados →
  el corpus puede construirse desde **datos estructurados** en lugar de
  scraping de texto completo (vía éticamente más limpia).
- Campos fiables por artículo: `claim_reviewed` (texto de la afirmación) y
  `verdict` (etiqueta). `claimant` y `tags` **no** son fiables en el JSON-LD.
- Distribución de veredictos fuertemente sesgada hacia `Falso` → el
  **desbalance de clases** será el reto central del modelo.

## Estructura

```
.
├── colombiacheck_recon.py   # Fase 1: scraper de reconocimiento (existente)
├── src/fake_news_co/        # paquete: adquisición, features, modelo
├── data/raw/                # datos crudos (ignorado por git)
├── data/processed/          # corpus limpio (ignorado por git)
├── models/                  # checkpoints (ignorado por git)
├── notebooks/               # exploración (EDA)
├── app/                     # demo Streamlit (Fase 4)
├── docs/                    # MODEL_CARD · DATASHEET · DATA_STATEMENT
└── tests/
```

## Uso

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                 # deps base (recon + datos)
pip install -e ".[nlp,app,dev]"  # stack completo (modelo + demo + dev)

# Reconocimiento
python colombiacheck_recon.py --max-pages 5   # smoke test
python colombiacheck_recon.py                 # archivo completo
```

## Ética y procedencia

- El scraper respeta `robots.txt` (`/chequeos` está permitido), se identifica en
  el `User-Agent`, cachea localmente y aplica *throttling* de 1.5 s.
- El corpus prioriza **datos estructurados públicos (ClaimReview)** sobre la
  cosecha de cuerpos completos. Cualquier uso de texto completo requiere contacto
  previo con ColombiaCheck (`contacto@colombiacheck.com`).
- Detalles completos en el [Datasheet](docs/DATASHEET.md).

## Licencia

Código bajo licencia MIT. Los textos de los chequeos son propiedad de
ColombiaCheck y **no** se redistribuyen en este repositorio.
