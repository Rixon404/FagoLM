---
base_model: Genereux-akotenou/BacteriaTIS-DNABERT-K6-89M
library_name: peft
tags:
- base_model:adapter:Genereux-akotenou/BacteriaTIS-DNABERT-K6-89M
- lora
- transformers
- dnabert
- bacteriophage
- tis-prediction
language:
- es
- en
---

# FagoLM-TIS (LoRA)

Adaptador **LoRA** (`r = 16`, `lora_alpha = 32`, `lora_dropout = 0.05`, capas objetivo `["query", "key", "value", "dense"]`) sobre [`Genereux-akotenou/BacteriaTIS-DNABERT-K6-89M`](https://huggingface.co/Genereux-akotenou/BacteriaTIS-DNABERT-K6-89M) para la identificación de sitios reales de inicio de traducción (TIS) y la discriminación frente a codones internos falsos en regiones truncadas en 5' (`open_5` y `open_both`) en genomas de bacteriófagos.

* **Autor:** Rixon René Riverol Arévalo — Maestría en Optimización y Cómputo Aplicado (MOCA), UAEM.
* **Repositorio del proyecto:** [https://github.com/Rixon404/FagoLM](https://github.com/Rixon404/FagoLM)

## Detalles del Entrenamiento

* **Parámetros entrenables:** `2,680,322` de `91,874,308` (`2.9174%`).
* **Tokenización y preprocesamiento:** Ventanas de `60` pb (`±30` pb alrededor del codón candidato) con padding `'N'` seguro en bordes `< 30` pb, tokenizadas en `6-mers` con paso `overlap = 1`.
* **Datos de entrenamiento / validación:** `28,000` / `5,000` ventanas balanceadas (`ATG`, `GTG`, `TTG`, Tabla 11), incorporando negativos duros provenientes de codones internos dentro de fragmentos `open_5` y `open_both` para evitar el recorte destructivo del borde físico 5'.
* **Hardware y consumo:** GPU NVIDIA RTX 3050 Ti (4 GB), `1,517.49 MiB` VRAM pico, `426.61 s`.

## Resultados Principales

* **Validación (`5,000` ventanas):** `Accuracy = 0.9548`, `Precision = 0.9698`, `Recall = 0.9388`, `F1 = 0.9541`, `FPR = 0.0292`, `AUROC = 0.9875`.
* **Exactitud sobre negativos duros en bordes 5' abiertos:** `97.03%` en `neg_open5_partial_5prime` y `98.17%` en `neg_open5_partial_both`.
* **Impacto conjunto con `FagoLM-CDS` y calibración en Test 3B (`1,301` fragmentos):** Eleva `Partial Edge Recovery` de `0.0000` (`e2_core_original`) y `0.4720` (`e2_abiertos_gt100`) a **`0.7874`** global (alcanzando **`0.8793`** en `partial_5prime > 103` nt y **`0.8710`** en `partial_both`).

## Carga con `peft`

```python
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

base_id = "Genereux-akotenou/BacteriaTIS-DNABERT-K6-89M"
adapter_dir = "etapa3_finetuning_lora/modelos/fagolm_tis_lora"

tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
model = PeftModel.from_pretrained(
    AutoModelForSequenceClassification.from_pretrained(base_id),
    adapter_dir,
).merge_and_unload().eval()
```
