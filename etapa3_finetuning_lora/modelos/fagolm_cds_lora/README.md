---
base_model: Genereux-akotenou/BacteriaCDS-DNABERT-K6-89M
library_name: peft
tags:
- base_model:adapter:Genereux-akotenou/BacteriaCDS-DNABERT-K6-89M
- lora
- transformers
- dnabert
- bacteriophage
- gene-prediction
language:
- es
- en
---

# FagoLM-CDS (LoRA)

Adaptador **LoRA** (`r = 16`, `lora_alpha = 32`, `lora_dropout = 0.05`, capas objetivo `["query", "key", "value", "dense"]`) sobre [`Genereux-akotenou/BacteriaCDS-DNABERT-K6-89M`](https://huggingface.co/Genereux-akotenou/BacteriaCDS-DNABERT-K6-89M) para la clasificación de regiones codificantes (CDS) en **genomas de bacteriófagos completos y fragmentos parciales/truncados** (`complete`, `open_5`, `open_3` y `open_both`, longitud `≥ 60` nt).

* **Autor:** Rixon René Riverol Arévalo — Maestría en Optimización y Cómputo Aplicado (MOCA), UAEM.
* **Repositorio del proyecto:** [https://github.com/Rixon404/FagoLM](https://github.com/Rixon404/FagoLM)

## Detalles del Entrenamiento

* **Parámetros entrenables:** `2,680,322` de `91,874,308` (`2.9174%`).
* **Tokenización:** `6-mers` con paso `overlap = 3` (últimos `510` nt de cada candidato, padding dinámico en FP16).
* **Datos de entrenamiento / validación:** `28,000` / `5,000` secuencias balanceadas (`50%` positivas, `50%` negativas incluyendo negativos reales de `60–103` nt para eliminar el sesgo de longitud del modelo base) extraídas de `120` fagos Train y `20` fagos Val de INPHARED (`GenomesDB_Jan_2026`), desreplicados contra los `20` fagos Test mediante contención de 21-mers canónicos (`< 0.0303`).
* **Hardware y consumo:** GPU NVIDIA RTX 3050 Ti (4 GB), `596.8 MiB` VRAM pico, `1,185.0 s`.

## Resultados Principales

* **Validación (`5,000` secuencias):** `Accuracy = 0.9762`, `Precision = 0.9772`, `Recall = 0.9752`, `F1 = 0.9762`, `FPR = 0.0228`, `AUROC = 0.9958` (en `≤ 103 nt`: `F1 = 0.9145`, `FPR = 0.0413`, `AUROC = 0.9663`).
* **Test 3A (Benchmark 2B directo en el banco congelado de `20` fagos):**
  * `complete_gt103`: Recall `0.9446 → 0.9917`, F1 `0.9681 → 0.9890`.
  * `sin_5prime_50pct`: Recall `0.5970 → 0.9407` (`+34.37 pp`), F1 `0.7460 → 0.9640`.
  * `sin_3prime_50pct_slice3p`: Recall `0.7565 → 0.9649` (`+20.84 pp`), F1 `0.8605 → 0.9795`.
  * `sin_ambos_50pct`: Recall `0.5766 → 0.9449` (`+36.83 pp`), F1 `0.7304 → 0.9669`.
  * `truncado_corto_le100` (`75` nt): FPR `1.0000 → 0.0483` (`-95.17 pp`), AUROC `0.5505 → 0.9671`, F1 `0.6667 → 0.8889`.

## Carga con `peft`

```python
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

base_id = "Genereux-akotenou/BacteriaCDS-DNABERT-K6-89M"
adapter_dir = "etapa3_finetuning_lora/modelos/fagolm_cds_lora"

tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
model = PeftModel.from_pretrained(
    AutoModelForSequenceClassification.from_pretrained(base_id),
    adapter_dir,
).merge_and_unload().eval()
```
