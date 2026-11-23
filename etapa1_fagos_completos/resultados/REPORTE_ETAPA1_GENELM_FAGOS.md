# Reporte Experimental — Etapa 1: Evaluación de GeneLM en Genomas Completos de Fagos (Piloto INPHARED)

## 1. Configuración del Experimento Piloto
- **Corpus origen:** INPHARED Enero 2026 (`GenomesDB_Jan_2026.tar.gz`, `2,977` genomas mono-contig con `.fna` + `.gff` generado por `Prodigal:002006` + PHROGs; ver [resumen del corpus](../piloto_inphared/resumen_corpus_inphared.json) y [Figura 1](../../graficas_tesis/fig01_corpus_inphared.png)).
- **Muestra piloto:** `20` genomas estratificados por longitud (`<20 kb`: 4, `20–50 kb`: 6, `50–100 kb`: 6, `>=100 kb`: 4; semilla `42`), sumando `1,400,404` pb y `2,051` CDS de referencia (ver [manifiesto piloto](../piloto_inphared/manifiesto_piloto.tsv)).
- **Checkpoints evaluados (sin *fine-tuning*):** `BacteriaCDS-DNABERT-K6-89M` y `BacteriaTIS-DNABERT-K6-89M`, inferencia en GPU (`RTX 3050 Ti 4 GB`, FP16 verificado con 100% de concordancia frente a FP32 en el primer genoma).
- **Gráficas asociadas:** [Fig. 3: Modos y embudo](../../graficas_tesis/fig03_etapa1_modos_y_embudo.png) | [Fig. 4: Desgloses biológicos](../../graficas_tesis/fig04_etapa1_desgloses_biologicos.png) | [Fig. 5: Atajo ≤100 nt y OOD TIS](../../graficas_tesis/fig05_atajo_longitud_y_ood_tis.png)
- **Modos comparados:**
  - `modo_a_original`: Replica las reglas de `core.py` (`starts=['TTG','CTG','ATG','GTG']`, `length_coding > 100`, `seq[-510:]` en ambas hebras, recorte TIS sin guarda en bordes).
  - `modo_b_adaptado_fagos`: Adapta solo el wrapper (`starts=['ATG','GTG','TTG','ATT','ATC','ATA']`, `total_len >= 60`, padding `'N'` en ventanas TIS de bordes `<30 pb`).
  - `modo_c_diagnostico_hebra_menos`: Igual a `modo_a_original`, pero recorta `seq[:510]` en hebra `-` (como `01-Analyse-CDS.ipynb` celda 16 al entrenar `BacteriaCDS`).

## 2. Resultados Globales por Modo
| mode | refs | preds | precision_iou | recall_iou | f1_iou | stop_precision | stop_recall | stop_f1 | exact_recall | tis_acc_cond | fp_per_kb | elapsed_seconds | peak_vram_mib |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| modo_a_original | 2051 | 1957 | 0.9515 | 0.9078 | 0.9291 | 0.9785 | 0.9337 | 0.9556 | 0.8415 | 0.9013 | 0.0678 | 82.308 | 1011.03 |
| modo_b_adaptado_fagos | 2051 | 10395 | 0.182 | 0.9225 | 0.304 | 0.1872 | 0.9488 | 0.3127 | 0.8211 | 0.8654 | 6.0718 | 149.891 | 1011.03 |
| modo_c_diagnostico_hebra_menos | 2051 | 1952 | 0.9513 | 0.9054 | 0.9278 | 0.9785 | 0.9313 | 0.9543 | 0.8391 | 0.901 | 0.0678 | 82.655 | 1011.03 |

## 3. Embudo de Pérdidas por Etapa (Etapa 1 CDS vs Etapa 2 TIS)
| mode | refs | extractor_stop | pct_extractor_stop | stage1_cds_pos | pct_stage1_cds_pos | stage2_tis_pred1 | pct_stage2_tis_pred1 | final_stop_tp | pct_final_stop_tp | final_exact_tp | pct_final_exact_tp | stage1_precision | stage2_empty_win | stage2_trunc_win |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| modo_a_original | 2051 | 2044 | 99.66 | 1939 | 94.54 | 1854 | 90.39 | 1915 | 93.37 | 1726 | 84.15 | 0.9773 | 5 | 0 |
| modo_b_adaptado_fagos | 2051 | 2051 | 100.0 | 1951 | 95.12 | 1864 | 90.88 | 1946 | 94.88 | 1684 | 82.11 | 0.1453 | 0 | 0 |
| modo_c_diagnostico_hebra_menos | 2051 | 2044 | 99.66 | 1934 | 94.3 | 1849 | 90.15 | 1910 | 93.13 | 1721 | 83.91 | 0.9773 | 5 | 0 |

## 4. Desglose por Hebra (`+` vs `-`)
| mode | strand | refs | preds | stage1_recall | stop_precision | stop_recall | stop_f1 | exact_recall | tis_acc_cond |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| modo_a_original | + | 1238 | 1193 | 0.9532 | 0.979 | 0.9435 | 0.9609 | 0.8554 | 0.9067 |
| modo_a_original | - | 813 | 764 | 0.9336 | 0.9777 | 0.9188 | 0.9474 | 0.8204 | 0.8929 |
| modo_b_adaptado_fagos | + | 1238 | 5256 | 0.9612 | 0.2258 | 0.9588 | 0.3656 | 0.8401 | 0.8762 |
| modo_b_adaptado_fagos | - | 813 | 5139 | 0.936 | 0.1477 | 0.9336 | 0.255 | 0.7921 | 0.8485 |
| modo_c_diagnostico_hebra_menos | + | 1238 | 1193 | 0.9532 | 0.979 | 0.9435 | 0.9609 | 0.8554 | 0.9067 |
| modo_c_diagnostico_hebra_menos | - | 813 | 759 | 0.9274 | 0.9776 | 0.9127 | 0.944 | 0.8143 | 0.8922 |

## 5. Desglose por Longitud de CDS (nt)
| mode | length_bin_nt | refs | extractor_stop | stage1_recall | stop_recall | exact_recall | tis_acc_cond |
| --- | --- | --- | --- | --- | --- | --- | --- |
| modo_a_original | <=100 | 10 | 5 | 0.0 | 0.0 | 0.0 | 0.0 |
| modo_a_original | 101-300 | 698 | 696 | 0.8754 | 0.8596 | 0.788 | 0.9167 |
| modo_a_original | 301-900 | 970 | 970 | 0.9856 | 0.9722 | 0.8845 | 0.9099 |
| modo_a_original | >900 | 373 | 373 | 0.9973 | 0.9973 | 0.8525 | 0.8548 |
| modo_b_adaptado_fagos | <=100 | 10 | 10 | 0.3 | 0.3 | 0.3 | 1.0 |
| modo_b_adaptado_fagos | 101-300 | 698 | 698 | 0.8868 | 0.8825 | 0.7736 | 0.8766 |
| modo_b_adaptado_fagos | 301-900 | 970 | 970 | 0.9866 | 0.9845 | 0.8557 | 0.8691 |
| modo_b_adaptado_fagos | >900 | 373 | 373 | 0.9973 | 0.9973 | 0.8338 | 0.836 |
| modo_c_diagnostico_hebra_menos | <=100 | 10 | 5 | 0.0 | 0.0 | 0.0 | 0.0 |
| modo_c_diagnostico_hebra_menos | 101-300 | 698 | 696 | 0.8682 | 0.8524 | 0.7808 | 0.916 |
| modo_c_diagnostico_hebra_menos | 301-900 | 970 | 970 | 0.9856 | 0.9722 | 0.8845 | 0.9099 |
| modo_c_diagnostico_hebra_menos | >900 | 373 | 373 | 0.9973 | 0.9973 | 0.8525 | 0.8548 |

## 6. Desglose por Soporte Funcional PHROG
| mode | phrog_status | refs | stage1_recall | stop_recall | exact_recall | tis_acc_cond |
| --- | --- | --- | --- | --- | --- | --- |
| modo_a_original | with_phrog | 1442 | 0.9626 | 0.9508 | 0.8585 | 0.903 |
| modo_a_original | without_phrog | 609 | 0.9048 | 0.8933 | 0.8013 | 0.8971 |
| modo_b_adaptado_fagos | with_phrog | 1442 | 0.9674 | 0.9646 | 0.8377 | 0.8684 |
| modo_b_adaptado_fagos | without_phrog | 609 | 0.913 | 0.9113 | 0.7816 | 0.8577 |
| modo_c_diagnostico_hebra_menos | with_phrog | 1442 | 0.9605 | 0.9487 | 0.8564 | 0.9028 |
| modo_c_diagnostico_hebra_menos | without_phrog | 609 | 0.9015 | 0.89 | 0.798 | 0.8967 |

## 7. Desglose por Estrato de Tamaño Genómico
| mode | stratum | refs | preds | precision_iou | recall_iou | f1_iou | stop_f1 | exact_recall | tis_acc_cond |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| modo_a_original | lt_20kb | 43 | 41 | 0.878 | 0.8372 | 0.8571 | 0.881 | 0.7209 | 0.8378 |
| modo_a_original | 20kb_50kb | 342 | 316 | 0.9335 | 0.8626 | 0.8967 | 0.9362 | 0.8041 | 0.8929 |
| modo_a_original | 50kb_100kb | 620 | 610 | 0.9574 | 0.9419 | 0.9496 | 0.9675 | 0.8629 | 0.8992 |
| modo_a_original | ge_100kb | 1046 | 990 | 0.9566 | 0.9054 | 0.9303 | 0.9578 | 0.8461 | 0.9077 |
| modo_b_adaptado_fagos | lt_20kb | 43 | 163 | 0.227 | 0.8605 | 0.3592 | 0.3689 | 0.7209 | 0.8158 |
| modo_b_adaptado_fagos | 20kb_50kb | 342 | 1628 | 0.1843 | 0.8772 | 0.3046 | 0.3168 | 0.7778 | 0.8526 |
| modo_b_adaptado_fagos | 50kb_100kb | 620 | 1596 | 0.369 | 0.95 | 0.5316 | 0.5433 | 0.8435 | 0.8688 |
| modo_b_adaptado_fagos | ge_100kb | 1046 | 7008 | 0.1378 | 0.9235 | 0.2399 | 0.2468 | 0.826 | 0.8692 |
| modo_c_diagnostico_hebra_menos | lt_20kb | 43 | 41 | 0.878 | 0.8372 | 0.8571 | 0.881 | 0.7209 | 0.8378 |
| modo_c_diagnostico_hebra_menos | 20kb_50kb | 342 | 314 | 0.9331 | 0.8567 | 0.8933 | 0.9329 | 0.7982 | 0.8922 |
| modo_c_diagnostico_hebra_menos | 50kb_100kb | 620 | 606 | 0.9571 | 0.9355 | 0.9462 | 0.9641 | 0.8565 | 0.8985 |
| modo_c_diagnostico_hebra_menos | ge_100kb | 1046 | 991 | 0.9566 | 0.9063 | 0.9308 | 0.9583 | 0.847 | 0.9078 |

## 8. Conclusiones y Diagnóstico de la Etapa 1

1. **Desempeño base sólido en genomas fágicos completos (`modo_a_original`):** Sin realizar *fine-tuning*, GeneLM obtiene `Precision (IoU≥0.5) = 0.9515`, `Recall (IoU≥0.5) = 0.9078`, `F1 (IoU≥0.5) = 0.9291` (`Stop_F1 = 0.9556`, `Exact_Recall = 0.8415`, `TIS_acc_cond = 0.9013`) con `1,011.03 MiB` de VRAM pico y `82.31 s` totales (`~4.1 s/genoma`) en FP16. El desempeño es menor en CDS hipotéticas sin motivo PHROG (`Stop_R = 0.8933` vs `0.9508` con PHROG), en fagos `< 20 kb` (`F1_iou = 0.8571`, `Exact_R = 0.7209`), en CDS `101–300` nt (`Stop_R = 0.8596`) y en CDS `<= 100` nt (`0.0%` de recuperación).
2. **Colapso de precisión en `modo_b_adaptado_fagos` explicado por sesgo de longitud (*length shortcut*):** Bajar el filtro a `total_len >= 60` e incluir `ATT/ATC/ATA` elevó `Stop_Recall` a `0.9488`, pero desplomó `Stop_Precision` a `0.1872` (`FP/kb = 6.0718`). El análisis directo de `etapa1_cds_groups.tsv` reveló que `BacteriaCDS-DNABERT-K6-89M` clasificó **`11,425 / 11,425` (`100.00%`)** de los grupos con `coding_len <= 100` nt como `CDS = 1` (frente a `1,998 / 14,589 = 13.70%` en `> 100` nt), consecuencia directa de que `01-Analyse-CDS.ipynb` (celda 16) aplicó `length > 100` solo a los negativos de entrenamiento (`label = 0`).
3. **Sensibilidad OOD en Etapa 2 (`BacteriaTIS`) ante `ATT/ATC/ATA` y fallos de borde `< 30 pb`:** En `modo_b`, aun entre llamadas `> 103` nt totales, `BacteriaTIS` seleccionó `127` inicios no vistos en su entrenamiento (`61 ATC`, `39 ATT`, `27 ATA`), contribuyendo a reducir `TIS_acc_cond` de `0.9013` a `0.8654`. En contraste, el padding `'N'` en bordes `< 30 pb` eliminó las `5` referencias de borde afectadas en la tabla de embudo (`stage2_empty_win = 0`, correspondientes a `16` ventanas candidatas TIS vacías y `1` truncada en `modo_a_original`).
4. **Asimetría de hebra `-` (`modo_a` vs `modo_c`):** Recortar `[:510]` en hebra `-` (`modo_c`) redujo levemente el recall en hebra `-` (`Stop_R = 0.9127` vs `0.9188`; `Exact_R = 0.8143` vs `0.8204`), confirmando que debe mantenerse el recorte `[-510:]` consistente con `core.py` para inferencia estándar, aunque la brecha de ~2 pp entre hebra `+` y hebra `-` persiste.
