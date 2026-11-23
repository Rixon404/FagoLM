# Reporte Experimental — Etapa 2: Comportamiento de GeneLM en Datos Parciales / Truncados

## 1. Diseño Experimental de Etapa 2
- **Gráficas asociadas:** [Fig. 6: Composición de fragmentos 1–10 kb](../../graficas_tesis/fig06_etapa2_dataset_fragmentos.png) | [Fig. 7: Benchmark 2B Antes vs. Después](../../graficas_tesis/fig07_benchmark2b_antes_vs_despues_lora.png)
- **Benchmark 2B (Prueba controlada directa en `BacteriaCDS-DNABERT-K6-89M`):** Evalúa la degradación pura del clasificador ante cortes exactos en marco (`complete_gt103`, `sin_5prime_50pct`, `sin_3prime_50pct_slice3p`, `sin_3prime_50pct_slice5p`, `sin_ambos_50pct`, `truncado_corto_le100`) usando las `2,051` CDS reales de los 20 fagos piloto y `2,051` negativos completos muestreados.
- **Benchmark 2A (Pipeline end-to-end sobre fragmentos tipo *contig*):** Evalúa `1,301` fragmentos estratificados (`1,000`: 732, `2,000`: 306, `5,000`: 162, `10,000`: 101; total `3,164,000` pb y `5,767` referencias observables: `3,608 complete`, `1,019 partial_5prime`, `985 partial_3prime`, `155 partial_both`) en 3 modos:
  - `e2_core_original`: pipeline estándar `core.py` (`partial3=False`, `partial5=False`, `coding_len > 100`).
  - `e2_abiertos_gt100`: candidatos en 6 marcos (`complete`, `open_5`, `open_3`, `open_both`) alineados al marco de lectura con `in_frame_len > 100` nt y padding `'N'` en ventanas TIS `< 30` pb.
  - `e2_abiertos_ge60`: igual al anterior pero bajando el corte a `in_frame_len >= 60` nt.

## 2. Benchmark 2B — Degradación Pura de `BacteriaCDS` ante Truncamiento Controlado
| condition | n_pos | n_neg | accuracy | precision | recall | f1 | fpr | auroc | mean_prob_pos | mean_prob_neg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| complete_gt103 | 2039 | 2051 | 0.9689 | 0.9928 | 0.9446 | 0.9681 | 0.0068 | 0.9976 | 0.915 | 0.0087 |
| sin_5prime_50pct | 1737 | 1312 | 0.7684 | 0.9942 | 0.597 | 0.746 | 0.0046 | 0.9881 | 0.6039 | 0.0066 |
| sin_3prime_50pct_slice3p | 1737 | 1312 | 0.8603 | 0.9977 | 0.7565 | 0.8605 | 0.0023 | 0.9943 | 0.7364 | 0.0044 |
| sin_3prime_50pct_slice5p | 1737 | 1312 | 0.8606 | 0.9977 | 0.7571 | 0.8609 | 0.0023 | 0.9944 | 0.7377 | 0.0044 |
| sin_ambos_50pct | 1762 | 1326 | 0.7571 | 0.9961 | 0.5766 | 0.7304 | 0.003 | 0.988 | 0.5855 | 0.0055 |
| truncado_corto_le100 | 2051 | 2051 | 0.5 | 0.5 | 1.0 | 0.6667 | 1.0 | 0.5505 | 0.9998 | 0.9998 |

## 3. Benchmark 2A — Resultados Globales en Fragmentos (`1, 2, 5, 10 kb`)
| mode | refs | preds | tp | fp | fn | precision | recall | f1 | exact_observable_recall | partial_edge_recovery | fp_per_kb | elapsed_seconds | peak_vram_mib |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e2_core_original | 5767 | 3710 | 3363 | 347 | 2404 | 0.9065 | 0.5831 | 0.7097 | 0.5131 | 0.0 | 0.1097 | 164.58 | 1014.96 |
| e2_abiertos_gt100 | 5767 | 4979 | 4528 | 451 | 1239 | 0.9094 | 0.7852 | 0.8427 | 0.6893 | 0.472 | 0.1425 | 197.092 | 1014.96 |
| e2_abiertos_ge60 | 5767 | 15801 | 4731 | 11070 | 1036 | 0.2994 | 0.8204 | 0.4387 | 0.7118 | 0.5428 | 3.4987 | 319.447 | 1014.96 |

## 4. Desglose por Categoría de Truncamiento (Todas vs `observable_bp > 103`)
| mode | subset | truncation_status | refs | matched | recall | exact_recall | edge_refs | edge_recovered | partial_edge_recovery |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e2_core_original | all_refs | complete | 3608 | 3207 | 0.8889 | 0.8201 | 0 | 0 | NA |
| e2_core_original | all_refs | partial_5prime | 1019 | 156 | 0.1531 | 0.0 | 1019 | 0 | 0.0 |
| e2_core_original | all_refs | partial_3prime | 985 | 0 | 0.0 | 0.0 | 985 | 0 | 0.0 |
| e2_core_original | all_refs | partial_both | 155 | 0 | 0.0 | 0.0 | 155 | 0 | 0.0 |
| e2_core_original | observable_bp_gt103 | complete | 3582 | 3207 | 0.8953 | 0.8261 | 0 | 0 | NA |
| e2_core_original | observable_bp_gt103 | partial_5prime | 820 | 156 | 0.1902 | 0.0 | 820 | 0 | 0.0 |
| e2_core_original | observable_bp_gt103 | partial_3prime | 793 | 0 | 0.0 | 0.0 | 793 | 0 | 0.0 |
| e2_core_original | observable_bp_gt103 | partial_both | 155 | 0 | 0.0 | 0.0 | 155 | 0 | 0.0 |
| e2_abiertos_gt100 | all_refs | complete | 3608 | 3247 | 0.8999 | 0.8329 | 0 | 0 | NA |
| e2_abiertos_gt100 | all_refs | partial_5prime | 1019 | 497 | 0.4877 | 0.2924 | 1019 | 299 | 0.2934 |
| e2_abiertos_gt100 | all_refs | partial_3prime | 985 | 686 | 0.6964 | 0.6477 | 985 | 686 | 0.6964 |
| e2_abiertos_gt100 | all_refs | partial_both | 155 | 98 | 0.6323 | 0.2194 | 155 | 34 | 0.2194 |
| e2_abiertos_gt100 | observable_bp_gt103 | complete | 3582 | 3247 | 0.9065 | 0.8389 | 0 | 0 | NA |
| e2_abiertos_gt100 | observable_bp_gt103 | partial_5prime | 820 | 497 | 0.6061 | 0.3634 | 820 | 299 | 0.3646 |
| e2_abiertos_gt100 | observable_bp_gt103 | partial_3prime | 793 | 682 | 0.86 | 0.7995 | 793 | 682 | 0.86 |
| e2_abiertos_gt100 | observable_bp_gt103 | partial_both | 155 | 98 | 0.6323 | 0.2194 | 155 | 34 | 0.2194 |
| e2_abiertos_ge60 | all_refs | complete | 3608 | 3275 | 0.9077 | 0.8362 | 0 | 0 | NA |
| e2_abiertos_ge60 | all_refs | partial_5prime | 1019 | 609 | 0.5976 | 0.3611 | 1019 | 394 | 0.3867 |
| e2_abiertos_ge60 | all_refs | partial_3prime | 985 | 749 | 0.7604 | 0.6964 | 985 | 744 | 0.7553 |
| e2_abiertos_ge60 | all_refs | partial_both | 155 | 98 | 0.6323 | 0.2194 | 155 | 34 | 0.2194 |
| e2_abiertos_ge60 | observable_bp_gt103 | complete | 3582 | 3263 | 0.9109 | 0.8389 | 0 | 0 | NA |
| e2_abiertos_ge60 | observable_bp_gt103 | partial_5prime | 820 | 522 | 0.6366 | 0.3646 | 820 | 319 | 0.389 |
| e2_abiertos_ge60 | observable_bp_gt103 | partial_3prime | 793 | 692 | 0.8726 | 0.7995 | 793 | 687 | 0.8663 |
| e2_abiertos_ge60 | observable_bp_gt103 | partial_both | 155 | 98 | 0.6323 | 0.2194 | 155 | 34 | 0.2194 |

## 5. Desglose por Longitud de Fragmento
| mode | length | fragments | refs | preds | precision | recall | f1 | exact_recall | partial_edge_recovery | fp_per_kb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e2_core_original | 1000 | 732 | 1696 | 617 | 0.8574 | 0.3119 | 0.4574 | 0.2441 | 0.0 | 0.1202 |
| e2_core_original | 2000 | 306 | 1178 | 689 | 0.8911 | 0.5212 | 0.6577 | 0.4525 | 0.0 | 0.1225 |
| e2_core_original | 5000 | 162 | 1327 | 1034 | 0.9149 | 0.7129 | 0.8014 | 0.6405 | 0.0 | 0.1086 |
| e2_core_original | 10000 | 101 | 1566 | 1370 | 0.9299 | 0.8135 | 0.8678 | 0.742 | 0.0 | 0.095 |
| e2_abiertos_gt100 | 1000 | 732 | 1696 | 1325 | 0.8777 | 0.6857 | 0.7699 | 0.5637 | 0.4661 | 0.2213 |
| e2_abiertos_gt100 | 2000 | 306 | 1178 | 995 | 0.8995 | 0.7598 | 0.8237 | 0.6638 | 0.471 | 0.1634 |
| e2_abiertos_gt100 | 5000 | 162 | 1327 | 1199 | 0.9249 | 0.8357 | 0.8781 | 0.7543 | 0.5106 | 0.1111 |
| e2_abiertos_gt100 | 10000 | 101 | 1566 | 1460 | 0.9322 | 0.8691 | 0.8995 | 0.7893 | 0.452 | 0.098 |
| e2_abiertos_ge60 | 1000 | 732 | 1696 | 4222 | 0.3015 | 0.7506 | 0.4302 | 0.6038 | 0.5399 | 4.0287 |
| e2_abiertos_ge60 | 2000 | 306 | 1178 | 3149 | 0.2998 | 0.8014 | 0.4363 | 0.6919 | 0.5458 | 3.6029 |
| e2_abiertos_ge60 | 5000 | 162 | 1327 | 3751 | 0.3029 | 0.8561 | 0.4474 | 0.7679 | 0.5816 | 3.2284 |
| e2_abiertos_ge60 | 10000 | 101 | 1566 | 4679 | 0.2945 | 0.8799 | 0.4413 | 0.7963 | 0.4915 | 3.2683 |

## 6. Diagnóstico por Etapa en Referencias Parciales (`observable_bp > 103`)
| mode | truncation_status | refs | stage1_cand_covered | pct_stage1_cand | stage1_pos_covered | pct_stage1_pos | final_matched | pct_final_matched | edge_recovered | pct_edge_recovered |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e2_core_original | complete | 3582 | 3497 | 97.63 | 3243 | 90.54 | 3207 | 89.53 | 0 | NA |
| e2_core_original | partial_5prime | 820 | 755 | 92.07 | 595 | 72.56 | 156 | 19.02 | 0 | 0.0 |
| e2_core_original | partial_3prime | 793 | 102 | 12.86 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 |
| e2_core_original | partial_both | 155 | 11 | 7.1 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 |
| e2_abiertos_gt100 | complete | 3582 | 3492 | 97.49 | 3240 | 90.45 | 3247 | 90.65 | 0 | NA |
| e2_abiertos_gt100 | partial_5prime | 820 | 819 | 99.88 | 648 | 79.02 | 497 | 60.61 | 299 | 36.46 |
| e2_abiertos_gt100 | partial_3prime | 793 | 772 | 97.35 | 668 | 84.24 | 682 | 86.0 | 682 | 86.0 |
| e2_abiertos_gt100 | partial_both | 155 | 155 | 100.0 | 152 | 98.06 | 98 | 63.23 | 34 | 21.94 |
| e2_abiertos_ge60 | complete | 3582 | 3496 | 97.6 | 3273 | 91.37 | 3263 | 91.09 | 0 | NA |
| e2_abiertos_ge60 | partial_5prime | 820 | 820 | 100.0 | 682 | 83.17 | 522 | 63.66 | 319 | 38.9 |
| e2_abiertos_ge60 | partial_3prime | 793 | 777 | 97.98 | 692 | 87.26 | 692 | 87.26 | 687 | 86.63 |
| e2_abiertos_ge60 | partial_both | 155 | 155 | 100.0 | 152 | 98.06 | 98 | 63.23 | 34 | 21.94 |

## 7. Diagnóstico Integrado y Cuellos de Botella

1. **`e2_core_original` falla masivamente en contigs cortos por diseño:** Con `partial3=False` y `partial5=False`, obtiene `Recall = 0.0` en `partial_3prime` (`0/985`) y `partial_both` (`0/155`), y apenas `0.1531` en `partial_5prime` (`0.0` recuperación de bordes), desplomando su `F1` a `0.4574` en fragmentos de `1 kb` y acumulando `432` ventanas TIS vacías en bordes `< 30` pb.
2. **Solo desbloquear candidatos abiertos (`e2_abiertos_gt100`) mejora `+13.30 pp` el F1 global (`0.7097 → 0.8427`) y `+31.25 pp` en `1 kb` (`0.4574 → 0.7699`):** Mantiene intacta la precisión (`0.9094`) y recupera el `86.00%` de las referencias `partial_3prime` con `observable_bp > 103`.
3. **Dos mecanismos distintos explican los errores residuales (objetivo directo de la Etapa 3):**
   - **Descalibración de `BacteriaCDS` sin fine-tuning (Benchmark 2B):** Aunque el `AUROC` se mantiene en `0.9880–0.9943`, las probabilidades de positivos truncados bajan (`mean_prob_pos = 0.5855` en `sin_ambos_50pct` y `0.6039` en `sin_5prime_50pct`), causando falsos negativos al umbral `0.5`.
   - **Sobre-llamado interno de `BacteriaTIS` sobre `open_5` y `open_both`:** `BacteriaCDS` sí acepta el `79.02%` de las `partial_5prime` y el `98.06%` de las `partial_both` (`> 103` nt), pero `BacteriaTIS` elige un codón de inicio interno falso en más de la mitad de los `open_5` y en `~78%` de los `open_both`, reduciendo `partial_edge_recovery` final a `36.46%` y `21.94%`, respectivamente.
4. **Sesgo `<= 100` nt codificantes (`truncado_corto_le100` y `e2_abiertos_ge60`):** Bajar el corte a `>= 60` nt en `e2_abiertos_ge60` desploma la precisión de `0.9094` a `0.2994` (`11,070` FP, `FP/kb = 3.4987`), consistente con el `FPR = 1.0000` (`AUROC = 0.5505`) observado en el Benchmark 2B sobre fragmentos internos de `75` nt.
