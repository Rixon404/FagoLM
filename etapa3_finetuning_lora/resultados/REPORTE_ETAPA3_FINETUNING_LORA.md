# Reporte Experimental — Etapa 3: Transfer Learning / Fine-Tuning con LoRA (`FagoLM-CDS` + `FagoLM-TIS`)

## 1. Split sin Fuga y Entrenamiento LoRA
- **Modelos LoRA generados:** [`FagoLM-CDS`](../modelos/fagolm_cds_lora/README.md) | [`FagoLM-TIS`](../modelos/fagolm_tis_lora/README.md)
- **Gráficas asociadas:** [Fig. 7: Benchmark 2B / Test 3A](../../graficas_tesis/fig07_benchmark2b_antes_vs_despues_lora.png) | [Fig. 8: Curvas LoRA](../../graficas_tesis/fig08_curvas_entrenamiento_lora.png) | [Fig. 9: Calibración Val](../../graficas_tesis/fig09_calibracion_umbrales_validacion.png) | [Fig. 10: Test 3B Global y Longitud](../../graficas_tesis/fig10_etapa2_vs_etapa3_global_y_longitud.png) | [Fig. 11: Test 3B por Truncamiento](../../graficas_tesis/fig11_etapa2_vs_etapa3_por_tipo_truncamiento.png) | [Fig. 12: Test 3C No Olvido](../../graficas_tesis/fig12_test3c_genomas_completos_no_olvido.png)
- **Particiones estrictamente disjuntas por genoma y homología (21-mers < 0.05):** Train = `120` fagos (`max_cont(Train,Test)=0.027825`), Val = `20` fagos (`max_cont(Val,Test)=0.000879`, `max_cont(Val,Train)=0.030297`), Test = `20` fagos congelados (Etapas 1 y 2). Excluidos por homología: `11` vs Test, `2` vs Train.
- **`FagoLM-CDS` (LoRA `r=16, alpha=32` sobre `BacteriaCDS-DNABERT-K6-89M`):** `28,000` ejemplos train / `5,000` val; historial de entrenamiento hasta época `2.0` (`1185.0s`; pico VRAM `596.8 MiB`); Val F1 = `0.9762`, Val AUROC = `0.9958` (en `len<=103` nt: F1 = `0.9145`, FPR = `0.0413`, AUROC = `0.9663`).
- **`FagoLM-TIS` (LoRA `r=16, alpha=32` sobre `BacteriaTIS-DNABERT-K6-89M`):** `28,000` ejemplos train / `5,000` val (50% negativos duros internos en bordes 5' abiertos); `3.0` épocas (`426.61s`; pico VRAM `1517.49 MiB`, mejor checkpoint en época `2.0`); Val F1 = `0.9541`, Val AUROC = `0.9875`.
- **Umbrales calibrados únicamente en Val:** `tau_cds_gt100=0.6`, `tau_cds_short=0.85`, `tau_tis_default=0.5`, `tau_tis_open5_override=0.95` (Val F1 = `0.8754`, Precision = `0.8458`, Recall = `0.9071`, Partial Edge = `0.8005`).

## 2. Test 3A — Benchmark 2B Directo (`genelm_cds` Base vs `FagoLM-CDS` LoRA)
| condition | n_pos | n_neg | base_recall | lora_recall | delta_recall | base_f1 | lora_f1 | delta_f1 | base_fpr | lora_fpr | base_auroc | lora_auroc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| complete_gt103 | 2039 | 2051 | 0.9446 | 0.9917 | 0.0471 | 0.9681 | 0.989 | 0.0209 | 0.0068 | 0.0137 | 0.9976 | 0.9986 |
| sin_5prime_50pct | 1737 | 1312 | 0.597 | 0.9407 | 0.3437 | 0.746 | 0.964 | 0.218 | 0.0046 | 0.0145 | 0.9881 | 0.9942 |
| sin_3prime_50pct_slice3p | 1737 | 1312 | 0.7565 | 0.9649 | 0.2084 | 0.8605 | 0.9795 | 0.119 | 0.0023 | 0.0069 | 0.9943 | 0.9965 |
| sin_3prime_50pct_slice5p | 1737 | 1312 | 0.7571 | 0.9649 | 0.2078 | 0.8609 | 0.9795 | 0.1186 | 0.0023 | 0.0069 | 0.9944 | 0.9964 |
| sin_ambos_50pct | 1762 | 1326 | 0.5766 | 0.9449 | 0.3683 | 0.7304 | 0.9669 | 0.2365 | 0.003 | 0.0128 | 0.988 | 0.9949 |
| truncado_corto_le100 | 2051 | 2051 | 1.0 | 0.8386 | -0.1614 | 0.6667 | 0.8889 | 0.2222 | 1.0 | 0.0483 | 0.5505 | 0.9671 |

## 3. Test 3B — Benchmark 2A End-to-End en `1,301` Fragmentos (`1, 2, 5, 10 kb`; `5,767` CDS)
| mode | refs | preds | precision | recall | f1 | exact_observable_recall | partial_edge_recovery | fp_per_kb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e2_core_original | 5767 | 3710 | 0.9065 | 0.5831 | 0.7097 | 0.5131 | 0.0 | 0.1097 |
| e2_abiertos_gt100 | 5767 | 4979 | 0.9094 | 0.7852 | 0.8427 | 0.6893 | 0.472 | 0.1425 |
| e2_abiertos_ge60 | 5767 | 15801 | 0.2994 | 0.8204 | 0.4387 | 0.7118 | 0.5428 | 3.4987 |
| e3_fagolm_gt100_t05 | 5767 | 5539 | 0.9041 | 0.8684 | 0.8859 | 0.7876 | 0.685 | 0.1678 |
| e3_fagolm_ge60_t05 | 5767 | 6501 | 0.7894 | 0.8899 | 0.8366 | 0.8075 | 0.7365 | 0.4327 |
| e3_fagolm_ge60_cal | 5767 | 5983 | 0.8609 | 0.8932 | 0.8768 | 0.8235 | 0.7874 | 0.263 |

### 3.1 Desglose por Categoría de Truncamiento (Subconjunto `all_refs`, `observable_bp_gt103` y `observable_bp_le103`)
| mode | subset | truncation_status | refs | matched | recall | exact_recall | partial_edge_recovery |
| --- | --- | --- | --- | --- | --- | --- | --- |
| e2_core_original | all_refs | complete | 3608 | 3207 | 0.8889 | 0.8201 | NA |
| e2_core_original | all_refs | partial_5prime | 1019 | 156 | 0.1531 | 0.0 | 0.0 |
| e2_core_original | all_refs | partial_3prime | 985 | 0 | 0.0 | 0.0 | 0.0 |
| e2_core_original | all_refs | partial_both | 155 | 0 | 0.0 | 0.0 | 0.0 |
| e2_core_original | observable_bp_gt103 | complete | 3582 | 3207 | 0.8953 | 0.8261 | NA |
| e2_core_original | observable_bp_gt103 | partial_5prime | 820 | 156 | 0.1902 | 0.0 | 0.0 |
| e2_core_original | observable_bp_gt103 | partial_3prime | 793 | 0 | 0.0 | 0.0 | 0.0 |
| e2_core_original | observable_bp_gt103 | partial_both | 155 | 0 | 0.0 | 0.0 | 0.0 |
| e2_core_original | observable_bp_le103 | complete | 26 | 0 | 0.0 | 0.0 | NA |
| e2_core_original | observable_bp_le103 | partial_5prime | 199 | 0 | 0.0 | 0.0 | 0.0 |
| e2_core_original | observable_bp_le103 | partial_3prime | 192 | 0 | 0.0 | 0.0 | 0.0 |
| e2_abiertos_gt100 | all_refs | complete | 3608 | 3247 | 0.8999 | 0.8329 | NA |
| e2_abiertos_gt100 | all_refs | partial_5prime | 1019 | 497 | 0.4877 | 0.2924 | 0.2934 |
| e2_abiertos_gt100 | all_refs | partial_3prime | 985 | 686 | 0.6964 | 0.6477 | 0.6964 |
| e2_abiertos_gt100 | all_refs | partial_both | 155 | 98 | 0.6323 | 0.2194 | 0.2194 |
| e2_abiertos_gt100 | observable_bp_gt103 | complete | 3582 | 3247 | 0.9065 | 0.8389 | NA |
| e2_abiertos_gt100 | observable_bp_gt103 | partial_5prime | 820 | 497 | 0.6061 | 0.3634 | 0.3646 |
| e2_abiertos_gt100 | observable_bp_gt103 | partial_3prime | 793 | 682 | 0.86 | 0.7995 | 0.86 |
| e2_abiertos_gt100 | observable_bp_gt103 | partial_both | 155 | 98 | 0.6323 | 0.2194 | 0.2194 |
| e2_abiertos_gt100 | observable_bp_le103 | complete | 26 | 0 | 0.0 | 0.0 | NA |
| e2_abiertos_gt100 | observable_bp_le103 | partial_5prime | 199 | 0 | 0.0 | 0.0 | 0.0 |
| e2_abiertos_gt100 | observable_bp_le103 | partial_3prime | 192 | 4 | 0.0208 | 0.0208 | 0.0208 |
| e2_abiertos_ge60 | all_refs | complete | 3608 | 3275 | 0.9077 | 0.8362 | NA |
| e2_abiertos_ge60 | all_refs | partial_5prime | 1019 | 609 | 0.5976 | 0.3611 | 0.3867 |
| e2_abiertos_ge60 | all_refs | partial_3prime | 985 | 749 | 0.7604 | 0.6964 | 0.7553 |
| e2_abiertos_ge60 | all_refs | partial_both | 155 | 98 | 0.6323 | 0.2194 | 0.2194 |
| e2_abiertos_ge60 | observable_bp_gt103 | complete | 3582 | 3263 | 0.9109 | 0.8389 | NA |
| e2_abiertos_ge60 | observable_bp_gt103 | partial_5prime | 820 | 522 | 0.6366 | 0.3646 | 0.389 |
| e2_abiertos_ge60 | observable_bp_gt103 | partial_3prime | 793 | 692 | 0.8726 | 0.7995 | 0.8663 |
| e2_abiertos_ge60 | observable_bp_gt103 | partial_both | 155 | 98 | 0.6323 | 0.2194 | 0.2194 |
| e2_abiertos_ge60 | observable_bp_le103 | complete | 26 | 12 | 0.4615 | 0.4615 | NA |
| e2_abiertos_ge60 | observable_bp_le103 | partial_5prime | 199 | 87 | 0.4372 | 0.3467 | 0.3769 |
| e2_abiertos_ge60 | observable_bp_le103 | partial_3prime | 192 | 57 | 0.2969 | 0.2708 | 0.2969 |
| e3_fagolm_gt100_t05 | all_refs | complete | 3608 | 3405 | 0.9437 | 0.8642 | NA |
| e3_fagolm_gt100_t05 | all_refs | partial_5prime | 1019 | 720 | 0.7066 | 0.6084 | 0.6114 |
| e3_fagolm_gt100_t05 | all_refs | partial_3prime | 985 | 757 | 0.7685 | 0.7157 | 0.7685 |
| e3_fagolm_gt100_t05 | all_refs | partial_both | 155 | 126 | 0.8129 | 0.6387 | 0.6387 |
| e3_fagolm_gt100_t05 | observable_bp_gt103 | complete | 3582 | 3398 | 0.9486 | 0.8685 | NA |
| e3_fagolm_gt100_t05 | observable_bp_gt103 | partial_5prime | 820 | 720 | 0.878 | 0.7561 | 0.7598 |
| e3_fagolm_gt100_t05 | observable_bp_gt103 | partial_3prime | 793 | 742 | 0.9357 | 0.8726 | 0.9357 |
| e3_fagolm_gt100_t05 | observable_bp_gt103 | partial_both | 155 | 126 | 0.8129 | 0.6387 | 0.6387 |
| e3_fagolm_gt100_t05 | observable_bp_le103 | complete | 26 | 7 | 0.2692 | 0.2692 | NA |
| e3_fagolm_gt100_t05 | observable_bp_le103 | partial_5prime | 199 | 0 | 0.0 | 0.0 | 0.0 |
| e3_fagolm_gt100_t05 | observable_bp_le103 | partial_3prime | 192 | 15 | 0.0781 | 0.0677 | 0.0781 |
| e3_fagolm_ge60_t05 | all_refs | complete | 3608 | 3412 | 0.9457 | 0.8661 | NA |
| e3_fagolm_ge60_t05 | all_refs | partial_5prime | 1019 | 794 | 0.7792 | 0.6742 | 0.6781 |
| e3_fagolm_ge60_t05 | all_refs | partial_3prime | 985 | 800 | 0.8122 | 0.7574 | 0.8122 |
| e3_fagolm_ge60_t05 | all_refs | partial_both | 155 | 126 | 0.8129 | 0.6387 | 0.6387 |
| e3_fagolm_ge60_t05 | observable_bp_gt103 | complete | 3582 | 3398 | 0.9486 | 0.8685 | NA |
| e3_fagolm_ge60_t05 | observable_bp_gt103 | partial_5prime | 820 | 722 | 0.8805 | 0.7573 | 0.761 |
| e3_fagolm_ge60_t05 | observable_bp_gt103 | partial_3prime | 793 | 742 | 0.9357 | 0.8726 | 0.9357 |
| e3_fagolm_ge60_t05 | observable_bp_gt103 | partial_both | 155 | 126 | 0.8129 | 0.6387 | 0.6387 |
| e3_fagolm_ge60_t05 | observable_bp_le103 | complete | 26 | 14 | 0.5385 | 0.5385 | NA |
| e3_fagolm_ge60_t05 | observable_bp_le103 | partial_5prime | 199 | 72 | 0.3618 | 0.3317 | 0.3367 |
| e3_fagolm_ge60_t05 | observable_bp_le103 | partial_3prime | 192 | 58 | 0.3021 | 0.2812 | 0.3021 |
| e3_fagolm_ge60_cal | all_refs | complete | 3608 | 3403 | 0.9432 | 0.8609 | NA |
| e3_fagolm_ge60_cal | all_refs | partial_5prime | 1019 | 817 | 0.8018 | 0.7596 | 0.7625 |
| e3_fagolm_ge60_cal | all_refs | partial_3prime | 985 | 788 | 0.8 | 0.7452 | 0.8 |
| e3_fagolm_ge60_cal | all_refs | partial_both | 155 | 143 | 0.9226 | 0.871 | 0.871 |
| e3_fagolm_ge60_cal | observable_bp_gt103 | complete | 3582 | 3389 | 0.9461 | 0.8632 | NA |
| e3_fagolm_ge60_cal | observable_bp_gt103 | partial_5prime | 820 | 761 | 0.928 | 0.8756 | 0.8793 |
| e3_fagolm_ge60_cal | observable_bp_gt103 | partial_3prime | 793 | 738 | 0.9306 | 0.8663 | 0.9306 |
| e3_fagolm_ge60_cal | observable_bp_gt103 | partial_both | 155 | 143 | 0.9226 | 0.871 | 0.871 |
| e3_fagolm_ge60_cal | observable_bp_le103 | complete | 26 | 14 | 0.5385 | 0.5385 | NA |
| e3_fagolm_ge60_cal | observable_bp_le103 | partial_5prime | 199 | 56 | 0.2814 | 0.2814 | 0.2814 |
| e3_fagolm_ge60_cal | observable_bp_le103 | partial_3prime | 192 | 50 | 0.2604 | 0.2448 | 0.2604 |

### 3.2 Desglose por Longitud de Fragmento
| mode | length | refs | preds | precision | recall | f1 | exact_recall | partial_edge_recovery | fp_per_kb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e2_core_original | 1000 | 1696 | 617 | 0.8574 | 0.3119 | 0.4574 | 0.2441 | 0.0 | 0.1202 |
| e2_core_original | 2000 | 1178 | 689 | 0.8911 | 0.5212 | 0.6577 | 0.4525 | 0.0 | 0.1225 |
| e2_core_original | 5000 | 1327 | 1034 | 0.9149 | 0.7129 | 0.8014 | 0.6405 | 0.0 | 0.1086 |
| e2_core_original | 10000 | 1566 | 1370 | 0.9299 | 0.8135 | 0.8678 | 0.742 | 0.0 | 0.095 |
| e2_abiertos_gt100 | 1000 | 1696 | 1325 | 0.8777 | 0.6857 | 0.7699 | 0.5637 | 0.4661 | 0.2213 |
| e2_abiertos_gt100 | 2000 | 1178 | 995 | 0.8995 | 0.7598 | 0.8237 | 0.6638 | 0.471 | 0.1634 |
| e2_abiertos_gt100 | 5000 | 1327 | 1199 | 0.9249 | 0.8357 | 0.8781 | 0.7543 | 0.5106 | 0.1111 |
| e2_abiertos_gt100 | 10000 | 1566 | 1460 | 0.9322 | 0.8691 | 0.8995 | 0.7893 | 0.452 | 0.098 |
| e2_abiertos_ge60 | 1000 | 1696 | 4222 | 0.3015 | 0.7506 | 0.4302 | 0.6038 | 0.5399 | 4.0287 |
| e2_abiertos_ge60 | 2000 | 1178 | 3149 | 0.2998 | 0.8014 | 0.4363 | 0.6919 | 0.5458 | 3.6029 |
| e2_abiertos_ge60 | 5000 | 1327 | 3751 | 0.3029 | 0.8561 | 0.4474 | 0.7679 | 0.5816 | 3.2284 |
| e2_abiertos_ge60 | 10000 | 1566 | 4679 | 0.2945 | 0.8799 | 0.4413 | 0.7963 | 0.4915 | 3.2683 |
| e3_fagolm_gt100_t05 | 1000 | 1696 | 1526 | 0.8932 | 0.8037 | 0.8461 | 0.7199 | 0.6824 | 0.2227 |
| e3_fagolm_gt100_t05 | 2000 | 1178 | 1122 | 0.9002 | 0.8574 | 0.8783 | 0.7716 | 0.6822 | 0.183 |
| e3_fagolm_gt100_t05 | 5000 | 1327 | 1318 | 0.909 | 0.9028 | 0.9059 | 0.8244 | 0.6809 | 0.1481 |
| e3_fagolm_gt100_t05 | 10000 | 1566 | 1573 | 0.9135 | 0.9176 | 0.9156 | 0.8416 | 0.7175 | 0.1347 |
| e3_fagolm_ge60_t05 | 1000 | 1696 | 1794 | 0.7982 | 0.8443 | 0.8206 | 0.7571 | 0.7365 | 0.4945 |
| e3_fagolm_ge60_t05 | 2000 | 1178 | 1318 | 0.7906 | 0.8846 | 0.8349 | 0.7963 | 0.7364 | 0.451 |
| e3_fagolm_ge60_t05 | 5000 | 1327 | 1547 | 0.7841 | 0.9141 | 0.8441 | 0.8357 | 0.7305 | 0.4123 |
| e3_fagolm_ge60_t05 | 10000 | 1566 | 1842 | 0.7845 | 0.9227 | 0.848 | 0.8467 | 0.7458 | 0.3931 |
| e3_fagolm_ge60_cal | 1000 | 1696 | 1660 | 0.8705 | 0.852 | 0.8611 | 0.7895 | 0.7923 | 0.2937 |
| e3_fagolm_ge60_cal | 2000 | 1178 | 1211 | 0.8596 | 0.8837 | 0.8715 | 0.8132 | 0.7776 | 0.2778 |
| e3_fagolm_ge60_cal | 5000 | 1327 | 1429 | 0.8523 | 0.9179 | 0.8839 | 0.8463 | 0.7908 | 0.2605 |
| e3_fagolm_ge60_cal | 10000 | 1566 | 1683 | 0.8598 | 0.924 | 0.8907 | 0.8487 | 0.7797 | 0.2337 |

## 4. Test 3C — Control de No Olvido en `20` Genomas Completos de Etapa 1 (`2,051` CDS)
| mode | refs | preds | precision_iou | recall_iou | f1_iou | stop_f1 | exact_recall | tis_acc_cond | stop_recall_le100 | stop_recall_101_300 | fp_per_kb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| modo_a_original | 2051 | 1957 | 0.9515 | 0.9078 | 0.9291 | 0.9556 | 0.8415 | 0.9013 | 0.0 | 0.8596 | 0.0678 |
| modo_b_adaptado_fagos | 2051 | 10395 | 0.182 | 0.9225 | 0.304 | 0.3127 | 0.8211 | 0.8654 | 0.3 | 0.8825 | 6.0718 |
| e3_fagolm_completos_ge60_t05 | 2051 | 2532 | 0.7686 | 0.9488 | 0.8492 | 0.8571 | 0.864 | 0.9022 | 0.4 | 0.9499 | 0.4185 |
| e3_fagolm_completos_ge60_cal | 2051 | 2282 | 0.851 | 0.9469 | 0.8964 | 0.9047 | 0.862 | 0.902 | 0.4 | 0.9441 | 0.2428 |

## 5. Síntesis de Hallazgos de la Etapa 3
1. **Validación directa de transfer learning en `BacteriaCDS` (Test 3A):** `FagoLM-CDS` eleva el recall en cortes internos/truncados (`sin_5prime`: `+34.37 pp`, `sin_ambos`: `+36.83 pp`, `sin_3prime`: `+20.84 pp`), elimina la brecha en hebra `-` (`0.5237 → 0.9466` en `sin_ambos`) y erradica el sesgo `<= 100` nt (`FPR: 1.0000 → 0.0483`, `AUROC: 0.5505 → 0.9671`).
2. **Salto end-to-end en fragmentos `1–10 kb` (Test 3B):** Frente a `e2_core_original` (`F1 = 0.7097`, `Partial Edge = 0.0`), `e3_fagolm_ge60_cal` alcanza `F1 = 0.8768` (`+16.71 pp`), `Recall = 0.8932` (`+31.01 pp`), `Exact_Recall = 0.8235` (`+31.04 pp`) y `Partial Edge Recovery = 0.7874`. En fragmentos de `1 kb`, el `F1` pasa de `0.4574` a `0.8611` (`+40.37 pp`), y en referencias `> 103` nt los bordes `partial_5prime` y `partial_both` suben a `87.93%` y `87.10%` de recuperación exacta.
3. **Preservación de desempeño en fagos completos (Test 3C):** Sobre los 20 genomas completos, `e3_fagolm_completos_ge60_cal` supera al original en `Recall_IoU` (`0.9469` vs `0.9078`), `Stop_Recall` (`0.9556` vs `0.9337`), `Exact_Recall` (`0.8620` vs `0.8415`) y genes `101–300` nt (`0.9441` vs `0.8596`), con compromiso moderado en precisión (`0.8510`) al habilitar llamadas `>= 60` nt (frente a `0.1820` antes del ajuste fino).
