# FagoLM — Predicción de Genes Codificantes en Genomas Completos y Parciales de Bacteriófagos mediante Transfer Learning sobre GeneLM

> **Repositorio oficial de tesis de maestría** | **Autor:** Rixon René Riverol Arévalo  
> **Programa:** Maestría en Optimización y Cómputo Aplicado (MOCA) — Facultad de Administración, Contaduría e Informática, **Universidad Autónoma del Estado de Morelos (UAEM)**, Cuernavaca, México.  
> **Comité Tutoral:** Dra. Lorena Díaz González (Asesora Interna), Dra. Blanca Itzelt Taboada Ramírez (Asesora Externa, IBT-UNAM), Dr. José Alberto Hernández Aguilar (Presidente) y Dr. Ayixon Sánchez Reyes (Vocal Externo, IBT-UNAM).

---

## 1. Descripción del Proyecto

**FagoLM** es una adaptación especializada del modelo fundacional de lenguaje genómico **GeneLM** (`BacteriaCDS-DNABERT-K6-89M` + `BacteriaTIS-DNABERT-K6-89M`) mediante **Transferencia de Aprendizaje eficiente en parámetros (LoRA)** y un **extractor abierto en 6 marcos de lectura alineado en fase codónica**, diseñada para predecir secuencias codificantes (CDS) y sitios de inicio de traducción (TIS) en **genomas de bacteriófagos completos y fragmentos parciales/truncados** (1–10 kb típicos de viromas metagenómicos).

### Problema que resuelve
1. El flujo original de **GeneLM** bloquea todo gen parcial (`partial3=False`, `partial5=False`, filtro `type:complete`), omite genes `≤ 103` nt (`length > 100`), contiene un fallo posicional silencioso en ventanas TIS cercanas a bordes (`< 30` pb) y arrastra un **atajo espurio de longitud** en `≤ 100` nt codificantes (`100.00%` de falsos positivos cuando se elimina el filtro de longitud sin reentrenar).
2. Aunque el codificador subyacente **DNABERT** conserva alta discriminabilidad interna en fragmentos truncados `> 103` nt (`AUROC ≥ 0.9880`), sus probabilidades se descalibran por pérdida de codón de parada o ausencia de contexto 5' y `BacteriaTIS` destruye hasta el **78%** de los bordes físicos `partial_5prime` / `partial_both` al sobre-llamar inicios internos falsos.
3. **FagoLM** corrige estas restricciones combinando extracción abierta (`complete`, `open_5`, `open_3`, `open_both`), padding seguro `'N'` en bordes `< 30` pb, ajuste fino LoRA (`r = 16, α = 32`; **2,680,322 parámetros entrenables = 2.9174%**) sobre `120` fagos disjuntos por homología (`21-mers < 0.0303`) y calibración multi-umbral sobre `20` fagos de validación independientes.

---

## 2. Resumen Cuantitativo de Resultados (Test Congelado: 20 Fagos / 1,301 Fragmentos)

### A. Etapa 1 & Test 3C — Genomas Completos (`20` fagos, `2,051` CDS, `1,400,404` pb)

| Modelo / Modo | Precisión (IoU≥0.5) | Recall (IoU≥0.5) | F1 (IoU≥0.5) | Stop F1 | Exact Recall | Recall 101–300 nt | Recall ≤100 nt | FP/kb |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `modo_a_original` (GeneLM estándar `>103 nt`) | **0.9515** | 0.9078 | **0.9291** | **0.9556** | 0.8415 | 0.8596 | 0.0000 | **0.0678** |
| `modo_b_adaptado_fagos` (`≥60 nt`, sin LoRA) | 0.1820 | 0.9225 | 0.3040 | 0.3127 | 0.8211 | 0.8825 | 0.3000 | 6.0718 |
| `e3_fagolm_completos_ge60_cal` (**FagoLM LoRA**) | 0.8510 | **0.9469** | 0.8964 | 0.9047 | **0.8620** | **0.9441** | **0.4000** | 0.2428 |

### B. Etapa 2 vs. Etapa 3 (Test 3B) — Fragmentos Simulados `1–10 kb` (`1,301` contigs, `5,767` CDS observables)

| Modelo / Modo | Precisión | Recall | F1 Global | F1 en `1 kb` | Exact Obs. Recall | Partial Edge Recovery | FP/kb |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e2_core_original` (GeneLM sin modificar) | 0.9065 | 0.5831 | 0.7097 | 0.4574 | 0.5131 | 0.0000 | **0.1097** |
| `e2_abiertos_gt100` (extractor abierto, sin LoRA) | **0.9094** | 0.7852 | 0.8427 | 0.7699 | 0.6893 | 0.4720 | 0.1425 |
| `e2_abiertos_ge60` (`≥60 nt`, sin LoRA) | 0.2994 | 0.8204 | 0.4387 | 0.4302 | 0.7118 | 0.5428 | 3.4987 |
| `e3_fagolm_gt100_t05` (**FagoLM LoRA `>100 nt`**) | 0.9041 | 0.8684 | **0.8859** | 0.8461 | 0.7876 | 0.6850 | 0.1678 |
| `e3_fagolm_ge60_cal` (**FagoLM LoRA Calibrado**) | 0.8609 | **0.8932** | 0.8768 | **0.8611** | **0.8235** | **0.7874** | 0.2630 |

* **Recuperación exacta de bordes truncados (`observable_bp > 103` nt en `e3_fagolm_ge60_cal`):**
  * `partial_5prime`: `0.1902 → 0.9280` Recall (`0.0000 → 0.8793` borde 5' exacto).
  * `partial_3prime`: `0.0000 → 0.9306` Recall (`0.0000 → 0.9306` borde 3' exacto).
  * `partial_both`: `0.0000 → 0.9226` Recall (`0.0000 → 0.8710` ambos bordes exactos).
* **Erradicación del atajo `≤ 100` nt (Benchmark 2B / Test 3A en fragmentos internos de `75` nt):** `FPR: 1.0000 → 0.0483`, `AUROC: 0.5505 → 0.9671`, `F1: 0.6667 → 0.8889`.

---

## 3. Estructura del Repositorio

```text
FagoLM/
├── README.md                                  # Documentación principal del proyecto
├── DEFICIENCIAS_Y_RESTRICCIONES_GENELM.md     # Auditoría técnica detallada de GeneLM (8 ejes)
├── documento_tesis/
│   └── Documento_Tesis_FagoLM_GeneLM_Fagos_Parciales.doc
├── etapa1_fagos_completos/
│   ├── piloto_inphared/                       # Manifiesto, resumen del corpus y genomas de prueba (20 fagos)
│   ├── predicciones/                          # Salidas GFF3/TSV por modo (Modos A, B y C)
│   └── resultados/                            # REPORTE_ETAPA1_GENELM_FAGOS.md + 7 tablas TSV
├── etapa2_datos_parciales/
│   ├── dataset_fragmentos/                    # Banco congelado de 1,301 fragmentos (1, 2, 5, 10 kb)
│   ├── benchmark_directo_cds/                 # Pruebas controladas de truncamiento directo (Benchmark 2B)
│   ├── predicciones/                          # Salidas por modo (e2_core_original, abiertos_gt100, abiertos_ge60)
│   └── resultados/                            # REPORTE_ETAPA2_DATOS_PARCIALES.md + 5 tablas TSV
├── etapa3_finetuning_lora/
│   ├── dataset_cds/ y dataset_tis/            # Datasets balanceados Train (28k) y Val (5k) sin fuga
│   ├── modelos/
│   │   ├── fagolm_cds_lora/                   # Adaptador LoRA FagoLM-CDS + tokenizer + métricas
│   │   └── fagolm_tis_lora/                   # Adaptador LoRA FagoLM-TIS + tokenizer + métricas
│   ├── calibracion_val/                       # Rejilla de 180 configuraciones y umbrales calibrados
│   ├── val_fragmentos/                        # 1,327 fragmentos independientes de validación
│   ├── predicciones/                          # Predicciones Test 3B y Test 3C
│   └── resultados/                            # REPORTE_ETAPA3_FINETUNING_LORA.md + 5 tablas TSV
├── graficas_tesis/                            # 12 figuras multipanel (PNG 300 DPI + PDF vectorial)
└── scripts/                                   # Pipeline reproducible completo (Etapas 1, 2 y 3)
```

---

## 4. Documentación, Reportes y Figuras

* **Auditoría técnica de GeneLM:** [`DEFICIENCIAS_Y_RESTRICCIONES_GENELM.md`](DEFICIENCIAS_Y_RESTRICCIONES_GENELM.md)
* **Reporte Etapa 1 (Fagos completos):** [`etapa1_fagos_completos/resultados/REPORTE_ETAPA1_GENELM_FAGOS.md`](etapa1_fagos_completos/resultados/REPORTE_ETAPA1_GENELM_FAGOS.md)
* **Reporte Etapa 2 (Datos parciales sin reentrenar):** [`etapa2_datos_parciales/resultados/REPORTE_ETAPA2_DATOS_PARCIALES.md`](etapa2_datos_parciales/resultados/REPORTE_ETAPA2_DATOS_PARCIALES.md)
* **Reporte Etapa 3 (Transfer Learning con LoRA):** [`etapa3_finetuning_lora/resultados/REPORTE_ETAPA3_FINETUNING_LORA.md`](etapa3_finetuning_lora/resultados/REPORTE_ETAPA3_FINETUNING_LORA.md)
* **Model Cards LoRA:**
  * [`FagoLM-CDS` (`r=16, α=32`)](etapa3_finetuning_lora/modelos/fagolm_cds_lora/README.md)
  * [`FagoLM-TIS` (`r=16, α=32`)](etapa3_finetuning_lora/modelos/fagolm_tis_lora/README.md)
* **Gráficas de tesis:** [`graficas_tesis/`](graficas_tesis) (`fig01_corpus_inphared` a `fig12_test3c_genomas_completos_no_olvido`).

---

## 5. Reproducibilidad del Pipeline (`scripts/`)

Todos los scripts resuelven sus rutas de entrada y salida de forma relativa al directorio raíz del repositorio (`FagoLM/`). Para ejecutar la tubería completa se requiere Python ≥ 3.11 con PyTorch + CUDA, `transformers`, `peft`, `accelerate`, `datasets`, `biopython`, `scikit-learn` y el fork [`Genereux-akotenou/orfipy`](https://github.com/Genereux-akotenou/orfipy) (necesario únicamente para `modo_a_original` / `e2_core_original`), así como los pesos base de GeneLM (`Genereux-akotenou/BacteriaCDS-DNABERT-K6-89M` y `BacteriaTIS-DNABERT-K6-89M`) y el archivo `GenomesDB_Jan_2026.tar.gz` de [INPHARED](https://github.com/RyanCook94/inphared) ubicados en `../trabajos_referencia/datasets/` (o especificados mediante `--models-dir` y `--tar-path`).

```bash
# Etapa 1: Piloto INPHARED y evaluación base
python scripts/preparar_piloto_inphared.py
python scripts/ejecutar_genelm_etapa1.py
python scripts/evaluar_etapa1.py

# Etapa 2: Generación de fragmentos parciales, Benchmark 2B y Benchmark 2A
python scripts/generar_dataset_parciales_etapa2.py
python scripts/evaluar_truncamiento_directo_cds.py
python scripts/ejecutar_genelm_etapa2.py
python scripts/evaluar_etapa2.py

# Etapa 3: Split libre de fuga (21-mers), Fine-Tuning LoRA, calibración Val y Tests 3A/3B/3C
python scripts/preparar_dataset_finetuning_etapa3.py
python scripts/entrenar_lora_fagolm.py --task all
python scripts/calibrar_umbrales_val_etapa3.py
python scripts/evaluar_etapa3_completa.py
```

---

## 6. Uso Rápido de los Adaptadores `FagoLM`

```python
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

base_cds = "Genereux-akotenou/BacteriaCDS-DNABERT-K6-89M"
adapter_cds = "etapa3_finetuning_lora/modelos/fagolm_cds_lora"

tokenizer = AutoTokenizer.from_pretrained(adapter_cds)
model = PeftModel.from_pretrained(
    AutoModelForSequenceClassification.from_pretrained(base_cds),
    adapter_cds,
).merge_and_unload().eval()
```

---

## 7. Referencias Base

* Akotenou, G., & El Allali, A. (2025). *Genomic language models (gLMs) decode bacterial genomes for improved gene prediction and translation initiation site identification*. Briefings in Bioinformatics, 26(4), bbaf311.
* Cook, R., et al. (2021). *INfrastructure for a PHAge REference Database (INPHARED)*. PHAGE, 2(4), 214–223.
* Hu, E. J., et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022.
