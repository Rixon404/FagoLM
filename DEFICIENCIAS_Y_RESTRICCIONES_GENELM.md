# Deficiencias, Limitaciones y Restricciones Técnicas de GeneLM para Genomas de Bacteriófagos (Completos y Parciales)

**Propósito del documento:** Registro técnico-metodológico verificado directamente sobre el código fuente, configuración de pesos, pipeline de entrenamiento y experimentos piloto locales de **GeneLM** (`Bioinformatics-UM6P/GeneLM`, commit `f05cabf`; modelos publicados en Hugging Face [`BacteriaCDS-DNABERT-K6-89M`](https://huggingface.co/Genereux-akotenou/BacteriaCDS-DNABERT-K6-89M) y [`BacteriaTIS-DNABERT-K6-89M`](https://huggingface.co/Genereux-akotenou/BacteriaTIS-DNABERT-K6-89M)), para sustentar la redacción del documento de tesis en [`Rixon404/FagoLM`](https://github.com/Rixon404/FagoLM) (marco metodológico, análisis de limitaciones del estado del arte, justificación de adaptaciones de preprocesamiento y estrategia de *fine-tuning* LoRA sobre genomas fágicos truncados).

> **Diagrama comparativo asociado:** [`graficas_tesis/fig02_arquitectura_y_restricciones_genelm.png`](graficas_tesis/fig02_arquitectura_y_restricciones_genelm.png)

---

## 1. Restricciones de Dominio y Arquitectura del Modelo

1. **Preentrenamiento y ajuste fino exclusivamente bacterianos (*domain shift*):**
   - GeneLM fue entrenado y evaluado sobre genomas bacterianos completos de RefSeq (`04_GeneLM/finetune/train-pipeline/data/cds/BacteriaCDS-ORFSeq2Class-v2/data_index.json` referencia `5,742` archivos bacterianos: `4,019` *train*, `1,148` *dev*, `575` *test*).
   - Los genomas de bacteriófagos presentan diferencias estructurales relevantes frente a bacterias: mayor densidad codificante (~90–95%), mayor frecuencia de genes cortos (< 150–200 pb), mayor proporción de genes solapados y acoplamiento traduccional, uso diferencial de codones y contenido GC/sesgo de codones adaptado al hospedero o divergente.
   - Al aplicar los pesos `BacteriaCDS-DNABERT-K6-89M` y `BacteriaTIS-DNABERT-K6-89M` sin adaptación, el modelo evalúa secuencias fágicas bajo la distribución estadística aprendida en bacterias.

2. **Arquitectura real verificada en los checkpoints (`genelm_cds` y `genelm_tis`):**
   - Aunque los archivos de configuración (`datasets/genelm_cds/config.json:27,32-34` y `datasets/genelm_tis/config.json:27,32-34`) contienen metadatos residuales `"rnn": "lstm"`, `"num_rnn_layer": 1`, `"rnn_hidden": 768`, la inspección directa de ambos `pytorch_model.bin` confirma que **no existen capas LSTM ni RNN** en los pesos publicados: ambos modelos contienen exactamente **202 tensores** correspondientes a un **DNABERT de 12 capas** (`hidden_size=768`, `num_attention_heads=12`, `max_position_embeddings=512`, `vocab_size=4101`) más una cabeza de clasificación lineal (`classifier.weight` de forma `[2, 768]`, `num_labels=2`).
   - En consecuencia, las predicciones dependen estrictamente de un límite máximo de **512 tokens** (equivalente a ~169 k-mers en Etapa 1 o hasta 510–515 nt según el paso de k-mers) y no cuentan con memoria recurrente para contextos largos.

3. **Representación fija en 6-mers y pérdida de información distal:**
   - **Etapa 1 (CDS):** Recorta únicamente los últimos `510` nucleótidos de cada ORF (`core.py:188,206`) y tokeniza en **6-mers con paso `overlap=3`** (`510` nt → 169 tokens + `[CLS]`/`[SEP]`). Cualquier señal en la región 5' de genes mayores a 510 nt queda completamente invisible para el clasificador de CDS.
   - **Etapa 2 (TIS):** Extrae una ventana fija de **60 pb** centrada en cada codón de inicio candidato (`30` pb aguas arriba y `30` pb aguas abajo, `core.py:210-248`) y tokeniza en **6-mers con paso `overlap=1`** (55 tokens + `[CLS]`/`[SEP]`). Señales fuera de esos ±30 pb no son evaluadas.

4. **Incapacidad estructural para modelar genes anidados en el mismo marco (mismo codón de parada):**
   - En `_cds_input_parser` (`core.py:177,195`), todos los ORFs que comparten el mismo codón de parada se agrupan y **solo el ORF más largo** (`max(group, key=lambda x: x['length'])`) se envía a `BacteriaCDS`.
   - En el post-procesamiento (`core.py:394-395`), para cada grupo clasificado como CDS (`orf_group`) se aplica `idxmax` sobre `prob_cls1` y se marca `prediction_max_likelihood = 1` únicamente en **un único TIS** por codón de parada. Genes fágicos con inicios alternativos funcionales o productos anidados en fase (ej. holina/antiholina o proteínas de lisis internas en mismo marco) no pueden recuperarse simultáneamente.

---

## 2. Restricciones Críticas para Genomas Parciales / Fragmentados (Metagenomas y Viromas)

Esta categoría representa el núcleo de incompatibilidad entre GeneLM estándar y el problema de predicción génica en *contigs* o genomas parciales:

1. **Bloqueo explícito de todo ORF parcial o truncado en el extractor (`core.py:79,347-348`):**
   - En la llamada a `orfipy_core.start_search` (`core.py:347-348`), GeneLM fija **`partial3=False`** (prohíbe ORFs sin codón de parada) y **`partial5=False`** (prohíbe ORFs sin codón de inicio).
   - Además, el parser `_parse_orfs` (`core.py:79`) aplica una expresión regular que exige literalmente `type:complete` en la cabecera del ORF.
   - **Impacto directo:** En fragmentos o *contigs* donde un gen está cortado en el extremo 5' (`partial_5prime`), en el extremo 3' (`partial_3prime`) o en ambos (`partial_both`), el pipeline estándar de GeneLM tiene **0% de capacidad estructural de predicción**, no por los pesos del transformer en sí, sino por el filtro previo de generación de candidatos.

2. **Dependencia implícita del extremo 3' y codón de parada en Etapa 1 (`core.py:188,206`):**
   - `BacteriaCDS` recibe `orf["sequence"][-510:]`, por lo que todos los ejemplos de entrada (cuando el gen mide ≥ 510 nt, o los más cortos completos) están **anclados en su extremo derecho al codón de parada** (`TAA`/`TAG`/`TGA`) en marco de lectura (`overlap=3`).
   - Cuando se extrae un candidato truncado en 3' (`open_3` o `open_both`, sin codón de parada), el anclaje derecho desaparece: la secuencia termina en el borde arbitrario del *contig* (que puede no ser múltiplo de 3 ni tener señal de terminación), generando un desplazamiento severo respecto a lo visto durante el entrenamiento bacteriano.

3. **Fallo silencioso y desalineación posicional en bordes genómicos `< 30 pb` durante Etapa 2 (`core.py:221,239`):**
   - En `_tis_input_parser` (`core.py:221`), en la hebra `+`, cuando un codón de inicio candidato se ubica a menos de 30 pb del extremo 5' del *contig* (`orf_start < 30`), la expresión Python `seq[orf_start - 30 : orf_start + 30]` combina un índice inicial negativo con un índice final positivo, por lo que **retorna una cadena vacía `""`** (0 pb).
   - En la hebra `-` (`core.py:239`), cuando el inicio del gen está a menos de 30 pb del extremo derecho del *contig* (`L - orf_end < 30`), la expresión `seq_rc[-(orf_end + 30) : -(orf_end - 30)]` recorta por izquierda a `0`, produciendo una ventana **truncada (< 60 pb) y desplazada** donde el codón de inicio ya no ocupa el centro (posición 30) sino la posición `L - orf_end`, rompiendo la alineación posicional esperada por `BacteriaTIS` (y si `orf_end <= 30`, también retorna `""`).
   - Las cadenas vacías generan 0 k-mers (`[CLS] [SEP]`) y emiten predicciones espurias sin advertencia. Evidencia adicional de este defecto en el propio flujo de los autores aparece en `train_tis_annotator.py:256`, donde debieron filtrar filas con `sequence` vacía convertida en `NaN` (`chunk[~chunk["sequence"].isna()]`).

4. **Umbral mínimo duro de longitud `length > 100` nt (`core.py:180,198`):**
   - El campo `length` reportado por `orfipy` corresponde a la longitud sin incluir el codón de parada (`end - start - 3`), de modo que `orf['length'] > 100` descarta todo ORF con longitud total `<= 103` nt incluyendo stop (≤ 33 aminoácidos; en la práctica, ≥ 105 nt totales / ≥ 102 nt codificantes) antes de pasar a `BacteriaCDS`.
   - Esto excluye automáticamente genes fágicos muy pequeños y fragmentos de genes truncados cerca de los bordes de un *contig* cuya porción observable mide ≤ 103 nt.

---

## 3. Restricciones Biológicas Específicas para Bacteriófagos

1. **Tabla genética fija en `table='1'` (`core.py:345` vs `run_single.py:38-43,122`):**
   - Aunque la interfaz CLI `run_single.py:38-43` expone el argumento `--codon_table` con valor por defecto `11` (*Bacterial/Archaeal*), **jamás lo transmite a `AnnotatorPipeline`**. El argumento solo se pasa al script auxiliar opcional `extract_orf_protein.py:122` tras concluir la anotación.
   - En `core.py:345`, `oc.start_search` recibe literalmente `'1'` (código genético estándar). Asimismo, GeneLM no contempla códigos genéticos alternativos relevantes en viromas fágicos (ej. tabla 15 de crAssphages / *Crassvirales* donde `TAG` se reasigna a glutamina, o tabla 4).

2. **Conjunto fijo de codones de inicio `['TTG', 'CTG', 'ATG', 'GTG']` (`core.py:343`):**
   - Incluye `CTG` (muy infrecuente como TIS real en bacterias/fagos, multiplicando falsos candidatos de inicio) y omite codones alternativos válidos de la tabla 11 (`ATT`, `ATC`, `ATA`).

3. **Rechazo fatal ante códigos de ambigüedad IUPAC (`core.py:319-325`):**
   - `AnnotatorPipeline.pipeline` valida que todos los caracteres de la secuencia pertenezcan a `set("ACGTUNacgtun")` y lanza `ValueError` si encuentra cualquier código de ambigüedad IUPAC (`R`, `Y`, `S`, `W`, `K`, `M`, `B`, `D`, `H`, `V`).
   - En el corpus completo INPHARED (`GenomesDB_Jan_2026.tar.gz`, `2,977` genomas) se comprobó que `113` genomas contienen `N` (aceptado por `ACGTUN`) y `0` contienen otros códigos IUPAC; sin embargo, en ensamblajes metagenómicos o contigs externos de viromas con ambigüedades no-`N`, esta comprobación provoca la cancelación completa del archivo FASTA.

---

## 4. Deficiencias de Ingeniería, Reproducibilidad y Pipeline de Fine-Tuning

1. **Dependencia de un *fork* no estándar de `orfipy` (`run-as-script/genelm/requirements.txt:3`):**
   - `core.py:336-352` llama a `orfipy_core.start_search` pasando **15 argumentos posicionales** (añadiendo `nested=True` en la posición 14, línea `350`).
   - La versión estándar de `orfipy` en PyPI (verificada en el entorno `base`) solo acepta **14 argumentos posicionales** y falla con `TypeError: start_search() takes exactly 14 positional arguments (15 given)`. Para correr `core.py` sin parchear es obligatorio compilar e instalar `git+https://github.com/Genereux-akotenou/orfipy`.

2. **Incompatibilidad sintáctica con Python < 3.12 (`run_single.py:62,105`):**
   - Las líneas `62` y `105` de `run_single.py` utilizan comillas dobles anidadas dentro de expresiones f-string (`f"...{'-'.join(task_uuid.split("-")[:2])}..."`), sintaxis introducida en PEP 701 (Python ≥ 3.12), provocando `SyntaxError: f-string: unmatched '('` en Python 3.10 y 3.11 (que son las versiones habituales de los entornos con PyTorch/CUDA).

3. **Acoplamiento de red, rutas relativas y efectos secundarios al importar (`core.py:17-18,27-31,287`):**
   - Al importar `genelm.core`, se ejecuta `Path("./__files__/results").mkdir(parents=True, exist_ok=True)` en el directorio de trabajo actual (`core.py:17-18`).
   - `AnnotatorPipeline.__init__` fija las rutas a HuggingFace Hub (`Genereux-akotenou/...`, `core.py:27-31`) sin aceptar parámetros de ruta local.
   - Al exportar GFF3 (`core.py:287`), escribe `"."` en la columna `phase` (en lugar de `"0"` para CDS completos en fase).

4. **Uso ineficiente de memoria GPU y cómputo en inferencia (`core.py:46-49,127,156`):**
   - `core.py:127,156` tokeniza siempre con **`padding="max_length"` hasta `512` tokens**, aun cuando las ventanas de Etapa 2 (TIS) miden apenas **55 tokens** (~9× desperdicio cuadrático de atención y VRAM) y las secuencias de Etapa 1 (CDS) como máximo **171 tokens** (~3× desperdicio).
   - `get_optimal_batch_size` (`core.py:46-49`) estima `free_memory // 15` (hasta 2048) sin activar FP16, pudiendo provocar desbordamiento de memoria (*CUDA OOM*) en GPUs de 4 GB (como la RTX 3050 Ti Laptop disponible).

5. **Asimetría de recorte en hebra `-` entre entrenamiento e inferencia (hallazgo crítico en `finetune/`):**
   - Al inspeccionar el script/notebook original de generación de datos de entrenamiento `04_GeneLM/finetune/train-pipeline/utils/01-Analyse-CDS.ipynb` (celda 16), se verificó que:
     - Para la hebra `+`, las secuencias de entrenamiento se recortaron con `longest_orf["sequence"][-max_seq_len:]` (últimos 510 nt, alineados al codón de parada).
     - Para la hebra `-`, se recortaron con `longest_orf["sequence"][:max_seq_len]` (**primeros 510 nt**, alineados al extremo 5'/inicio), lo cual se comprueba directamente en los ejemplos reales de `finetune/train-pipeline/data/verified-genome2/cds/balanced/bacteria_1.csv:3` (fila 3 empieza por `ATG...` y carece de codón de parada al final).
     - Adicionalmente, en `01-Analyse-CDS.ipynb` (celda 15), la regex de `parse_orfs` utiliza `frame:(\d+)` (sin `-?`), a diferencia de `core.py:79` que usa `frame:(-?\d+)`.
   - En contraste, el módulo de inferencia (`core.py:188,206`) recorta **`[-max_seq_len:]` en ambas hebras** (`+` y `-`). Esta inconsistencia entrenamiento-inferencia en la hebra negativa debe evaluarse empíricamente antes y durante cualquier *fine-tuning*.

6. **Documentación contradictoria y bugs en los scripts de *fine-tuning* (`finetune/`):**
   - `finetune/README.md:4` declara que los contenidos de `finetune/data-pipeline/*` y `finetune/train-pipeline/*` fueron excluidos de la publicación, aunque sí están presentes en el repositorio (`commit f05cabf`), incluyendo `train_cds_annotator.py` y `train_tis_annotator.py` con soporte nativo para **LoRA (`peft`)**.
   - Ambos scripts (`train_cds_annotator.py` y `train_tis_annotator.py`) presentan dos defectos menores reproducibles:
     1. Línea `71`: una coma final en `evaluation_strategy: str = field(default="steps"),` convierte el valor por defecto en una tupla `("steps",)` si no se sobrescribe explícitamente por CLI, y usa `evaluation_strategy` (renombrado a `eval_strategy` en versiones recientes de `transformers`).
     2. Líneas `397-400` (`train_cds_annotator.py`) / `398-401` (`train_tis_annotator.py`): invocan `torch.distributed.get_rank()` verificando solo `is_available()` pero no `is_initialized()`, por lo que fallan si no se ejecutan mediante `torchrun`.
   - Los archivos CSV procesados de los 5,742 genomas bacterianos originales no fueron publicados en el repositorio (solo `data_index.json`, `data_summary.json` y una muestra de `verified-genome2/`).

7. **Atajo de longitud (*length shortcut*) y fuga de etiquetas en el entrenamiento de `BacteriaCDS` (`01-Analyse-CDS.ipynb`, celda 16):**
   - Al generar los ejemplos de entrenamiento bacterianos en `01-Analyse-CDS.ipynb` (celda 16), el filtro `if orf['length'] > 100:` se aplicó **exclusivamente a la clase negativa (`label = 0`)** y nunca a los positivos reales (`label = 1`).
   - Como consecuencia, en el conjunto de entrenamiento no existió ningún ejemplo negativo con longitud codificante `<= 100` nt, por lo que `BacteriaCDS-DNABERT-K6-89M` aprendió un atajo degenerado: **clasificar el 100.00% de las secuencias cortas (`<= 100` nt codificantes) como `CDS = 1`**, lo que impide simplemente relajar el filtro de longitud de `core.py` sin reentrenar o hacer *fine-tuning* con negativos cortos reales (ver evidencia empírica en la Sección 6).

---

## 5. Consideraciones Metodológicas sobre el Dataset de Referencia INPHARED (`GenomesDB_Jan_2026.tar.gz`)

1. **Formato del archivo:** Aunque lleva extensión `.tar.gz`, `GenomesDB_Jan_2026.tar.gz` (10.6 GB) es un archivo **POSIX tar sin compresión gzip** (`2,977` directorios mono-contig con `.fna`, `.gff` y `.gbk`/`.gbf`).
2. **Naturaleza de las etiquetas de referencia en `.gff`:** Al escanear los **2,977 genomas completos** (`265,819` CDS totales), se verificó que **100%** de las anotaciones GFF provienen de **`Prodigal:002006`** (100% múltiplos de 3, 100% con codón de parada válido `TAA`/`TGA`/`TAG`, `len_le_60 = 0`, y únicamente inicios `ATG` [89.59%], `GTG` [7.85%] y `TTG` [2.56%]), enriquecidas con motivos HMM de **PHROGs** en el **`72.937%`** (`193,880 / 265,819`) de las CDS. Por ello:
   - Evaluar GeneLM contra estos GFF mide concordancia frente al estándar *de facto* Prodigal+PHROGs, no necesariamente certeza experimental absoluta en genes hipotéticos cortos.
   - Es metodológicamente indispensable reportar métricas diferenciando **CDS con respaldo funcional (`phrog_*`)** frente a **CDS hipotéticas sin motivo PHROG**.

---

## 6. Evidencia Empírica en Fagos Completos (Etapa 1 — Piloto INPHARED)

Se evaluaron los checkpoints originales de GeneLM (`datasets/genelm_cds` y `datasets/genelm_tis`, **sin *fine-tuning***) en GPU (`RTX 3050 Ti 4 GB`, FP16 verificado con `100%` de acuerdo respecto a FP32 en `PP511821`) sobre un piloto estratificado de **`20` genomas completos de INPHARED** (`1,400,404` pb, `2,051` CDS de referencia; `1,442` con PHROG y `609` sin PHROG; `seed=42`). Resultados completos en [`etapa1_fagos_completos/resultados/REPORTE_ETAPA1_GENELM_FAGOS.md`](etapa1_fagos_completos/resultados/REPORTE_ETAPA1_GENELM_FAGOS.md).

### 6.1 Desempeño global de GeneLM sin modificar (`modo_a_original`) vs variantes diagnósticas

| Modo experimental | Refs | Preds | Precision (IoU≥0.5) | Recall (IoU≥0.5) | F1 (IoU≥0.5) | Stop Precision | Stop Recall | Stop F1 | Exact Recall | TIS Acc Cond. | FP / kb | VRAM pico |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `modo_a_original` (reglas `core.py`) | 2,051 | 1,957 | **0.9515** | 0.9078 | **0.9291** | **0.9785** | 0.9337 | **0.9556** | **0.8415** | **0.9013** | **0.0678** | 1,011 MiB |
| `modo_b_adaptado_fagos` (tabla 11, `>=60 nt`, padding `N`) | 2,051 | 10,395 | 0.1820 | **0.9225** | 0.3040 | 0.1872 | **0.9488** | 0.3127 | 0.8211 | 0.8654 | 6.0718 | 1,011 MiB |
| `modo_c_diagnostico_hebra_menos` (`[:510]` en `-`) | 2,051 | 1,952 | 0.9513 | 0.9054 | 0.9278 | 0.9785 | 0.9313 | 0.9543 | 0.8391 | 0.9010 | 0.0678 | 1,011 MiB |

### 6.2 Hallazgos cuantitativos clave sobre las restricciones de GeneLM

1. **Desempeño base sólido en fagos completos, pero con brecha real en genes cortos, fagos pequeños y TIS:**
   - En genomas completos bajo `modo_a_original`, GeneLM alcanza un **Stop F1 = 0.9556** y **IoU F1 = 0.9291** (~4.1 s/genoma con padding dinámico en FP16, 1.01 GB VRAM), pero la **recuperación exacta de inicio (`Exact_Recall`) cae a 0.8415**:
     - Etapa 1 (`BacteriaCDS`) recupera el `94.54%` (`1,939 / 2,051`) de las paradas de CDS con `97.73%` de precisión sobre grupos candidatos.
     - Etapa 2 (`BacteriaTIS`) pierde ~9.2 puntos porcentuales entre parada y coordenada exacta (`TIS_acc_cond = 90.13%`), y descarta el `4.15%` de los inicios verdaderos (`stage2_tis_pred1 = 90.39%`, aunque parte se recupera cuando otro inicio cercano supera el umbral).
   - **Brecha por soporte PHROG:** En CDS con soporte PHROG (`1,442`), `Stop_Recall = 0.9508` y `Exact_Recall = 0.8585`; en CDS hipotéticas sin PHROG (`609`), baja a `Stop_Recall = 0.8933` y `Exact_Recall = 0.8013`.
   - **Brecha en genes cortos y genomas pequeños:**
     - Genes `<= 100` nt (`10` en el piloto; `1,136` de `<= 103` nt en todo INPHARED = `0.427%`): **0.0% de recuperación** en `modo_a_original` por el corte `orf['length'] > 100`.
     - Genes `101–300` nt (`698` en el piloto): `Stage1_Recall` cae a `0.8754`, `Stop_Recall` a `0.8596` y `Exact_Recall` a `0.7880` (vs. `98.56%`–`99.73%` en Etapa 1 para genes `> 300` nt).
     - Fagos pequeños (`< 20 kb`): `IoU F1 = 0.8571` y `Exact_Recall = 0.7209` (`tis_acc_cond = 0.8378`).
     - Genes largos (`> 900` nt): aunque `Stage1_Recall = 0.9973`, la precisión condicional de inicio cae a **`tis_acc_cond = 0.8548`** (en parte por la mayor cantidad de inicios alternativos internos `ATG/GTG/TTG/CTG` evaluados en el grupo).

2. **Confirmación empírica del atajo de longitud (*length shortcut*) en `BacteriaCDS` (`modo_b_adaptado_fagos`):**
   - Al bajar el umbral a `total_len >= 60` e incluir `ATT/ATC/ATA`, `cds_pred1` se disparó de `1,984` a **`13,423`** y las llamadas finales de `1,957` a **`10,395`** (`Stop_Precision` colapsó de `97.85%` a **`18.72%`**; `FP/kb = 6.0718`).
   - Al inspeccionar los `26,014` grupos evaluados en Etapa 1 en `modo_b`, se verificó que **`11,425 / 11,425` (`100.00%`)** de todos los grupos con `coding_len <= 100` nt fueron predichos como clase `1`, frente a `1,998 / 14,589` (`13.70%`) en los de `> 100` nt. Esto confirma experimentalmente la fuga de etiquetas heredada de `01-Analyse-CDS.ipynb` celda 16 (Sección 4.7).

3. **Degradación fuera de distribución (*OOD*) en `BacteriaTIS` con codones `ATT/ATC/ATA`:**
   - Incluso restringiendo las llamadas finales de `modo_b` al régimen `> 103` nt totales (`1,975` llamadas), `BacteriaTIS` eligió **`127` inicios no vistos en su entrenamiento** (`61 ATC`, `39 ATT`, `27 ATA`), contribuyendo a que `TIS_acc_cond` bajara de `90.13%` (`modo_a`) a `86.54%` (`modo_b`) y `Exact_Recall` de `84.15%` a `82.11%`, pese a que `Stop_Recall` subió de `93.37%` a `94.88%`.

4. **Impacto real del fallo en bordes `< 30 pb` (`stage2_empty_win`):**
   - En el piloto hubo `5` CDS de referencia con inicio a `< 30` pb del borde del genoma (`1,049` en todo el corpus INPHARED, `0.395%`). En `modo_a_original` se registraron **`16` ventanas TIS completamente vacías y `1` truncada** (afectando el 100% de las referencias de borde), mientras que en `modo_b_adaptado_fagos` el padding con `'N'` eliminó por completo las ventanas vacías/truncadas (`0`/`0`). En genomas fragmentados/parciales (Etapa 2), la fracción de genes truncados o cercanos al borde crece drásticamente, por lo que esta corrección resulta obligatoria.

5. **Resolución empírica de la asimetría de hebra `-` (`modo_a` vs `modo_c`):**
   - Invertir el recorte en hebra `-` a `[:510]` (`modo_c_diagnostico_hebra_menos`, tal como preparó los datos `01-Analyse-CDS.ipynb` celda 16) **no mejoró** el rendimiento respecto a `modo_a_original` (`[-510:]` en ambas hebras): en hebra `-`, `Stage1_Recall` bajó de `0.9336` a `0.9274`, `Stop_Recall` de `0.9188` a `0.9127` y `Exact_Recall` de `0.8204` a `0.8143`. No obstante, persiste una brecha constante de ~2–2.5 puntos porcentuales entre hebra `+` (`Stage1_R = 0.9532`, `Stop_R = 0.9435`) y hebra `-` (`Stage1_R = 0.9336`, `Stop_R = 0.9188`) que debe vigilarse en las Etapas 2 y 3.

### Implicaciones directas para las Etapas 2 (datos parciales) y 3 (*fine-tuning*)
- **Para Etapa 2 (fragmentos truncados sin 5', sin 3' o sin ambos):** el pipeline estándar de `core.py` (`partial3=False`, `partial5=False`) no puede detectar ningún gen truncado, y aun alimentando candidatos abiertos directos a `BacteriaCDS` + `BacteriaTIS`, habrá que: (a) **corregir las ventanas de borde `< 30` pb** (como en `modo_b`), y (b) **evaluar cuidadosamente el efecto del sesgo `<= 100` nt** cuando un fragmento corto truncado deje una porción observable `<= 100` nt codificantes.
- **Para Etapa 3 (*fine-tuning* con LoRA):** resulta indispensable incluir ejemplos negativos reales en todo el rango de longitudes cortas (`<= 100` nt) para eliminar el atajo de longitud, mantener los codones de inicio consistentes con la tabla 11 (`ATG/GTG/TTG`, evaluando aparte si conviene incluir inicios alternativos), y entrenar `BacteriaCDS` explícitamente sobre candidatos `open_5`, `open_3` y `open_both` carentes de codón de parada en 3'.

---

## 7. Evidencia Empírica en Datos Parciales / Truncados (Etapa 2)

Para cuantificar la degradación de GeneLM ante genomas fragmentados (`sin 5'`, `sin 3'` y `sin ambos`) sin realizar todavía *fine-tuning*, se ejecutaron dos benchmarks complementarios sobre los 20 genomas piloto de INPHARED (reporte completo en [`etapa2_datos_parciales/resultados/REPORTE_ETAPA2_DATOS_PARCIALES.md`](etapa2_datos_parciales/resultados/REPORTE_ETAPA2_DATOS_PARCIALES.md)):

### 7.1 Benchmark 2B: Prueba controlada directa sobre `BacteriaCDS-DNABERT-K6-89M`

| Condición de truncamiento en marco | Pos / Neg | Accuracy | Precision | Recall | F1 | FPR | AUROC | Prob. media (+) | Prob. media (-) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `complete_gt103` (control completo) | 2,039 / 2,051 | **0.9689** | 0.9928 | **0.9446** | **0.9681** | 0.0068 | **0.9976** | 0.9150 | 0.0087 |
| `sin_3prime_50pct_slice3p` (mitad 5', sin stop) | 1,737 / 1,312 | 0.8603 | 0.9977 | 0.7565 | 0.8605 | 0.0023 | 0.9943 | 0.7364 | 0.0044 |
| `sin_3prime_50pct_slice5p` (mitad 5', `[:510]`) | 1,737 / 1,312 | 0.8606 | 0.9977 | 0.7571 | 0.8609 | 0.0023 | 0.9944 | 0.7377 | 0.0044 |
| `sin_5prime_50pct` (mitad 3', con stop) | 1,737 / 1,312 | 0.7684 | 0.9942 | **0.5970** | 0.7460 | 0.0046 | **0.9881** | 0.6039 | 0.0066 |
| `sin_ambos_50pct` (interno, sin inicio ni stop) | 1,762 / 1,326 | 0.7571 | 0.9961 | **0.5766** | 0.7304 | 0.0030 | **0.9880** | 0.5855 | 0.0055 |
| `truncado_corto_le100` (75 nt internos) | 2,051 / 2,051 | **0.5000** | 0.5000 | 1.0000 | 0.6667 | **1.0000** | **0.5505** | **0.9998** | **0.9998** |

**Hallazgos críticos de Benchmark 2B:**
1. **Caída severa de recall por descalibración, pero representaciones internas preservadas (`AUROC >= 0.988`):** Al truncar las secuencias manteniendo el marco (`>= 105` nt), el recall cae **-18.8 pp** en `sin_3prime` (`0.7565`), **-34.8 pp** en `sin_5prime` (`0.5970`) y **-36.8 pp** en `sin_ambos` (`0.5766`, bajando hasta **`0.5237`** en hebra `-`). Sin embargo, el **AUROC permanece en `0.9880–0.9944`** y la precisión en `> 99.4%`: el modelo *sigue distinguiendo* ADN codificante de no codificante, pero sus probabilidades para positivos truncados caen de `0.9150` a `0.5855–0.7364`, quedando por debajo del umbral `0.5`. Esto demuestra empíricamente la **alta viabilidad del transfer learning / fine-tuning** sobre el backbone DNABERT.
2. **Colapso total bajo `<= 100` nt codificantes:** En fragmentos internos de `75` nt (`truncado_corto_le100`), el modelo emite probabilidad `0.9998` tanto para positivos como para negativos (`FPR = 1.0000`, `AUROC = 0.5505`), confirmando que el sesgo de longitud destruye por completo la discriminación por debajo de `103` nt.

### 7.2 Benchmark 2A: Pipeline completo sobre `1,301` fragmentos (`1, 2, 5, 10 kb`; `3.164 Mbp`, `5,767` CDS observables)

| Modo sobre fragmentos | Refs | Preds | Precision | Recall | F1 | Exact Obs. Recall | Partial Edge Rec. | FP / kb | Recall `complete` | Recall `partial_5'` | Recall `partial_3'` | Recall `partial_both` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e2_core_original` (`core.py`) | 5,767 | 3,710 | 0.9065 | **0.5831** | **0.7097** | 0.5131 | **0.0000** | 0.1097 | 0.8889 | 0.1531 | **0.0000** | **0.0000** |
| `e2_abiertos_gt100` (`>100 nt`, candidatos abiertos) | 5,767 | 4,979 | **0.9094** | **0.7852** | **0.8427** | **0.6893** | **0.4720** | 0.1425 | 0.8999 | 0.4877 | **0.6964** | 0.6323 |
| `e2_abiertos_ge60` (`>=60 nt`, candidatos abiertos) | 5,767 | 15,801 | **0.2994** | 0.8204 | 0.4387 | 0.7118 | 0.5428 | **3.4987** | 0.9077 | 0.5976 | 0.7604 | 0.6323 |

*(Desglose en el subconjunto `observable_bp > 103` nt para `e2_abiertos_gt100`: `complete` `0.9065`, `partial_5prime` `0.6061` [borde `0.3646`], `partial_3prime` **`0.8600`** [borde `0.8600`], `partial_both` `0.6323` [borde `0.2194`].)*

**Cuellos de botella verificados en fragmentos reales:**
1. **Bloqueo estructural del pipeline estándar (`e2_core_original`):** En fragmentos de `1 kb` (donde abundan genes truncados), el F1 de GeneLM sin adaptar cae a **`0.4574`** (`Recall = 0.3119`), con **0.0%** de recuperación en `partial_3prime` y `partial_both`, **0.0%** de recuperación de bordes parciales y **`432` ventanas TIS vacías** por el bug `< 30` pb.
2. **Ganancia sustancial solo con desbloqueo de candidatos abiertos (`e2_abiertos_gt100`):** Sin tocar los pesos, habilitar candidatos abiertos en marco + padding `'N'` en bordes TIS sube el F1 global **+13.30 pp** (`0.7097 → 0.8427`), y en `1 kb` **+31.25 pp** (`0.4574 → 0.7699`), sin perder precisión (`0.9094`).
3. **Segundo gran cuello de botella — `BacteriaTIS` destruye los bordes 5' abiertos (`partial_5prime` y `partial_both`):**
   - El diagnóstico por etapa sobre referencias `> 103` nt mostró que `BacteriaCDS` (`stage1_pos_covered`) sí cubre el **`79.02%`** (`648/820`) de las `partial_5prime` y el **`98.06%`** (`152/155`) de las `partial_both`.
   - Sin embargo, al pasar esos grupos a Etapa 2, `BacteriaTIS` suele dar `prob >= 0.5` a algún codón `ATG/GTG/TTG/CTG` interno falso y recorta el gen antes del borde 5': la recuperación exacta del borde físico cae de `79.02%` a **`36.46%`** en `partial_5prime` y de `98.06%` a **`21.94%`** en `partial_both` (y en muchos casos el gen queda tan acortado que cae por debajo de `IoU >= 0.5`).
4. **Imposibilidad de bajar de `103` nt sin reentrenar (`e2_abiertos_ge60`):** Incluir fragmentos parciales de `60–103` nt sube `partial_edge_recovery` a `0.5428`, pero dispara los falsos positivos a `11,070` (`FP/kb = 3.4987`, `Precision = 0.2994`).

### Requisitos técnicos derivados para la Etapa 3 (*Transfer Learning / Fine-Tuning* con LoRA)
1. **Reentrenar/calibrar `BacteriaCDS` con ejemplos truncados en marco (`open_5`, `open_3`, `open_both` y negativos reales cortos `60–103` nt)** para eliminar el atajo de longitud y recuperar los ~35–37 pp de recall perdidos en `sin_5prime` y `sin_ambos` aprovechando el `AUROC >= 0.988`.
2. **Entrenar o calibrar la decisión 5'-abierta en `BacteriaTIS` (o un clasificador conjunto de TIS vs continuación 5')** para distinguir entre un verdadero sitio de inicio interno y un codón `ATG/GTG/TTG` interno dentro de un gen que continúa más allá del borde 5' del *contig*.
3. **Mantener obligatoriamente la partición por genoma (sin fuga entre fragmentos de un mismo fago en *train/val/test*)**, la alineación estricta de marco de lectura (`frame` en `open_5`/`open_both`) y el padding seguro `'N'` en ventanas TIS cercanas a bordes.

---

## 8. Resultados del Transfer Learning / Fine-Tuning con LoRA (`FagoLM-CDS` + `FagoLM-TIS` — Etapa 3)

Se ejecutó el ajuste fino eficiente por parámetros (**LoRA**, `r=16, alpha=32`, `2,680,322` parámetros entrenables = `2.9174%`) sobre `genelm_cds` (`FagoLM-CDS`, `1,185 s`, `596.8 MiB` VRAM pico) y `genelm_tis` (`FagoLM-TIS`, `426.6 s`, `1,517.5 MiB` VRAM pico), utilizando un *split* estrictamente disjunto por genoma y homología (`21-mers` canónicos, contención máxima `< 0.0303`): `120` fagos Train / `20` fagos Validación / `20` fagos Test congelados (`28,000` ejemplos Train y `5,000` Val por tarea). Reporte completo en [`etapa3_finetuning_lora/resultados/REPORTE_ETAPA3_FINETUNING_LORA.md`](etapa3_finetuning_lora/resultados/REPORTE_ETAPA3_FINETUNING_LORA.md).

### 8.1 Deficiencias resueltas o mitigadas cuantitativamente (Antes vs. Después en Test congelado)

1. **Descalibración de `BacteriaCDS` ante genomas truncados (Test 3A — Benchmark 2B directo):**
   - `complete_gt103`: Recall `0.9446 → 0.9917` (`+4.71 pp`), F1 `0.9681 → 0.9890`, AUROC `0.9976 → 0.9986`.
   - `sin_3prime_50pct_slice3p`: Recall `0.7565 → 0.9649` (`+20.84 pp`), F1 `0.8605 → 0.9795`.
   - `sin_5prime_50pct`: Recall **`0.5970 → 0.9407` (`+34.37 pp`)**, F1 `0.7460 → 0.9640`.
   - `sin_ambos_50pct`: Recall **`0.5766 → 0.9449` (`+36.83 pp`)** (en hebra `-`: `0.5237 → 0.9466`, `+42.29 pp`, eliminando la brecha de hebra observada en Etapas 1 y 2), F1 `0.7304 → 0.9669`.
   - **Erradicación del *length shortcut* (`truncado_corto_le100`, `75` nt internos):** El FPR se desplomó de **`1.0000 → 0.0483` (`-95.17 pp`)**, la Precisión subió de `0.5000 → 0.9456`, el F1 de `0.6667 → 0.8889` y el AUROC de **`0.5505 → 0.9671` (`+0.4166`)**.

2. **Recuperación integral en fragmentos tipo *contig* `1/2/5/10 kb` (Test 3B — `1,301` fragmentos, `5,767` CDS):**
   - Frente al GeneLM original (`e2_core_original`: `P=0.9065`, `R=0.5831`, `F1=0.7097`, `Exact_R=0.5131`, `PartialEdge=0.0000`), **`e3_fagolm_ge60_cal`** (con umbrales calibrados únicamente en Val: `tau_cds_gt100=0.60`, `tau_cds_short=0.85`, `tau_tis_default=0.50`, `tau_tis_open5_override=0.95`) logra **`P=0.8609`, `R=0.8932` (`+31.01 pp`), `F1=0.8768` (`+16.71 pp`), `Exact_R=0.8235` (`+31.04 pp`) y `PartialEdge=0.7874` (`0% → 78.74%`)**.
   - Con el mismo corte `> 100` nt (`e3_fagolm_gt100_t05`), LoRA obtiene **`P=0.9041`, `R=0.8684`, `F1=0.8859`** (`+4.32 pp` en F1 y `+21.30 pp` en recuperación de bordes parciales sobre `e2_abiertos_gt100`).
   - En fragmentos de **`1 kb`** (el escenario más severo): `F1` salta de **`0.4574`** (`e2_core_original`) → **`0.7699`** (`e2_abiertos_gt100`) → **`0.8611`** (`e3_fagolm_ge60_cal`, **`+40.37 pp`**), y `Exact_Recall` de `0.2441` → **`0.7895` (`+54.54 pp`)**.
   - **Resolución del sobre-llamado interno de `BacteriaTIS` en bordes 5' abiertos (referencias `observable_bp > 103` nt):**
     - `partial_5prime`: Recall `0.1902` (`core`) → `0.6061` (`abiertos`) → **`0.9280`** (`e3_fagolm_ge60_cal`), y recuperación exacta del borde 5' **`0.0000 → 0.3646 → 0.8793`**.
     - `partial_3prime`: Recall `0.0000` (`core`) → `0.8600` (`abiertos`) → **`0.9306`**.
     - `partial_both`: Recall `0.0000` (`core`) → `0.6323` (`abiertos`) → **`0.9226`**, y recuperación exacta de ambos bordes **`0.0000 → 0.2194 → 0.8710`**.
   - Comparando 1-a-1 en régimen `>= 60` nt con umbral `0.50` (`e2_abiertos_ge60` vs `e3_fagolm_ge60_t05`), los falsos positivos bajaron de `11,070` (`3.4987 FP/kb`) a `1,369` (`0.4327 FP/kb`), elevando el F1 de `0.4387 → 0.8366`.

3. **Control de no olvido en genomas completos (Test 3C — `20` fagos completos, `2,051` CDS):**
   - `FagoLM-LoRA` no sufre olvido catastrófico en genes completos: en `e3_fagolm_completos_ge60_cal`, `Recall_IoU` sube de `0.9078 → 0.9469`, `Stop_Recall` de `0.9337 → 0.9556`, `Exact_Recall` de `0.8415 → 0.8620`, el recall en `101–300` nt sube de `0.8596 → 0.9441`, y en genes diminutos `<= 100` nt pasa de `0.0% → 40.0%` (`4/10`), reduciendo los `FP/kb` del régimen `>= 60` nt en **96%** respecto a `modo_b_adaptado_fagos` (`6.0718 → 0.2428 FP/kb`).

### 8.2 Limitaciones y compromisos estructurales que persisten tras el *fine-tuning* LoRA
1. **Compromiso precisión–cobertura en candidatos ultracortos (`60–103` nt):** Aunque el *length shortcut* fue eliminado (`AUROC = 0.9671`, `FPR = 0.0483`), en un escaneo exhaustivo de 6 marcos sobre genomas completos o fragmentos existen ~10× más ORFs aleatorios cortos no codificantes que genes reales `<= 103` nt. Por ello, activar el régimen `60–103` nt en genomas completos reduce la precisión de `0.9515` (`modo_a_original`, que omite todo `<= 103` nt) a `0.8510` (`e3_fagolm_completos_ge60_cal`), y en fragmentos truncados de `<= 103` nt observables la recuperación bajo calibración estricta (`tau_cds_short=0.85`) queda en `~28.8%` (`120/417` en Test 3B), lo que sugiere que fragmentos muy cortos carecen de suficiente contexto de 6-mers y podrían beneficiarse de modelos de codificación por entropía/frecuencias de codones complementarios o umbral adaptativo según longitud del *contig*.
2. **Restricciones arquitectónicas heredadas del wrapper/tokenización:** Siguen vigentes la imposibilidad de predecir genes anidados que comparten el mismo codón de parada (`idxmax` único por `orf_group`), la ausencia de soporte nativo para tablas genéticas alternativas (ej. tabla 15 de crAssphages) y el uso de `Prodigal:002006` + PHROGs como estándar de referencia en INPHARED (con las salvedades de la Sección 5).
