#!/usr/bin/env python3
"""Evalúa los resultados de la Etapa 2 (Benchmark 2A sobre fragmentos 1/2/5/10 kb
e integración con Benchmark 2B directo sobre BacteriaCDS) y genera reportes TSV/MD.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

_REPO_DIR = Path(__file__).resolve().parent.parent
_PARENT_DIR = _REPO_DIR.parent

DEFAULT_FRAG_DIR = (_REPO_DIR / "etapa2_datos_parciales/dataset_fragmentos"
)
DEFAULT_PRED_DIR = (_REPO_DIR / "etapa2_datos_parciales/predicciones"
)
DEFAULT_B2B_PATH = (_REPO_DIR / "etapa2_datos_parciales/benchmark_directo_cds/resultados_truncamiento_directo.tsv"
)
DEFAULT_OUT_DIR = (_REPO_DIR / "etapa2_datos_parciales/resultados"
)

MODES = ("e2_core_original", "e2_abiertos_gt100", "e2_abiertos_ge60")
STATUSES = ("complete", "partial_5prime", "partial_3prime", "partial_both")
FRAG_LENGTHS = (1000, 2000, 5000, 10000)


def read_tsv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def calc_f1(p: float, r: float) -> float:
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def parse_gff_preds(path: Path) -> dict[str, list[dict]]:
    by_frag: dict[str, list[dict]] = defaultdict(list)
    if not path.is_file():
        return by_frag
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw or raw.startswith("#"):
            if raw.strip() == "##FASTA":
                break
            continue
        parts = raw.split("\t")
        if len(parts) != 9:
            continue
        frag_id, _src, ftype, s1_s, e1_s, _sc, strand, _ph, attrs = parts
        if ftype.lower() not in {"cds", "orf"} or strand not in {"+", "-"}:
            continue
        attr_map = dict(item.split("=", 1) for item in attrs.split(";") if "=" in item)
        by_frag[frag_id].append(
            {
                "start0": int(s1_s) - 1,
                "end0": int(e1_s),
                "strand": strand,
                "cand_type": attr_map.get("cand_type", ""),
            }
        )
    return by_frag


def interval_iou(a0: int, a1: int, b0: int, b1: int) -> float:
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union else 0.0


def maximum_matching_iou(refs: list[dict], preds: list[dict], min_iou: float = 0.5):
    adj: list[list[int]] = []
    ious: dict[tuple[int, int], float] = {}
    for ri, r in enumerate(refs):
        cands = []
        for pi, p in enumerate(preds):
            if r["strand_sym"] != p["strand"]:
                continue
            iou = interval_iou(r["vis_start0"], r["vis_end0"], p["start0"], p["end0"])
            if iou >= min_iou:
                cands.append(pi)
                ious[(ri, pi)] = iou
        cands.sort(key=lambda pi: (-ious[(ri, pi)], pi))
        adj.append(cands)

    pred_to_ref: dict[int, int] = {}

    def augment(ri: int, seen: set[int]) -> bool:
        for pi in adj[ri]:
            if pi in seen:
                continue
            seen.add(pi)
            if pi not in pred_to_ref or augment(pred_to_ref[pi], seen):
                pred_to_ref[pi] = ri
                return True
        return False

    for ri in range(len(refs)):
        augment(ri, set())
    return {ri: pi for pi, ri in pred_to_ref.items()}, ious


def expected_edges(status: str, strand_sym: str) -> set[str]:
    if status == "partial_both":
        return {"left", "right"}
    if status == "partial_5prime":
        return {"left"} if strand_sym == "+" else {"right"}
    if status == "partial_3prime":
        return {"right"} if strand_sym == "+" else {"left"}
    return set()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frag-dir", type=Path, default=DEFAULT_FRAG_DIR)
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED_DIR)
    parser.add_argument("--b2b-path", type=Path, default=DEFAULT_B2B_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-iou", type=float, default=0.5)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    windows = read_tsv(args.frag_dir / "ventanas.tsv")
    raw_refs = read_tsv(args.frag_dir / "referencias_cds_observables.tsv")

    refs_by_frag: dict[str, list[dict]] = defaultdict(list)
    for r in raw_refs:
        strand_sym = "+" if str(r["strand"]) in {"1", "+"} else "-"
        refs_by_frag[r["fragment_id"]].append(
            {
                **r,
                "frag_len": int(r["fragment_length"]),
                "vis_start0": int(r["visible_start_local_0"]),
                "vis_end0": int(r["visible_end_local_0"]),
                "obs_bp": int(r["observable_bp"]),
                "strand_sym": strand_sym,
            }
        )

    summary_rows: list[dict] = []
    trunc_rows: list[dict] = []
    frag_len_rows: list[dict] = []
    matrix_rows: list[dict] = []
    diag_rows: list[dict] = []

    for mode in MODES:
        mode_dir = args.pred_dir / mode
        preds_by_frag = parse_gff_preds(mode_dir / "predicciones.gff")
        rt_path = mode_dir / "runtime_stats.json"
        rt = json.loads(rt_path.read_text(encoding="utf-8")) if rt_path.is_file() else {}

        # Agrupar etapa1_grupos.tsv por (fragment_id, strand)
        grp_by_frag_strand: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for g in read_tsv(mode_dir / "etapa1_grupos.tsv"):
            grp_by_frag_strand[(g["fragment_id"], g["strand"])].append(
                {
                    "start0": int(g["span_start0"]),
                    "end0": int(g["span_end0"]),
                    "cds_pred": int(g["cds_pred"]),
                    "region_kind": g["region_kind"],
                }
            )

        tot = Counter()
        len_acc = defaultdict(Counter)
        trunc_acc = defaultdict(Counter)  # (subset, status) -> Counter
        mat_acc = defaultdict(Counter)    # (frag_len, status) -> Counter
        diag_acc = defaultdict(Counter)   # (subset, status) -> Counter

        for w in windows:
            fid = w["fragment_id"]
            flen = int(w["length"])
            frefs = refs_by_frag.get(fid, [])
            fpreds = preds_by_frag.get(fid, [])
            ref_to_pred, _ = maximum_matching_iou(frefs, fpreds, min_iou=args.min_iou)

            tp = len(ref_to_pred)
            fp = len(fpreds) - tp
            fn = len(frefs) - tp
            assert tp + fn == len(frefs)
            assert tp + fp == len(fpreds)

            tot["fragments"] += 1
            tot["bp"] += flen
            tot["refs"] += len(frefs)
            tot["preds"] += len(fpreds)
            tot["tp"] += tp
            tot["fp"] += fp
            tot["fn"] += fn

            len_acc[flen]["fragments"] += 1
            len_acc[flen]["bp"] += flen
            len_acc[flen]["refs"] += len(frefs)
            len_acc[flen]["preds"] += len(fpreds)
            len_acc[flen]["tp"] += tp
            len_acc[flen]["fp"] += fp

            for ri, r in enumerate(frefs):
                st = r["truncation_status"]
                matched = int(ri in ref_to_pred)
                pred = fpreds[ref_to_pred[ri]] if matched else None
                exact = int(
                    matched
                    and pred["start0"] == r["vis_start0"]
                    and pred["end0"] == r["vis_end0"]
                )
                req_edges = expected_edges(st, r["strand_sym"])
                edge_rec = int(
                    matched
                    and bool(req_edges)
                    and ("left" not in req_edges or pred["start0"] == 0)
                    and ("right" not in req_edges or pred["end0"] == flen)
                )

                tot["exact"] += exact
                len_acc[flen]["exact"] += exact
                if req_edges:
                    tot["edge_refs"] += 1
                    tot["edge_rec"] += edge_rec
                    len_acc[flen]["edge_refs"] += 1
                    len_acc[flen]["edge_rec"] += edge_rec

                # Diagnóstico Etapa 1 vs Etapa 2
                cand_groups = grp_by_frag_strand.get((fid, r["strand_sym"]), [])
                s1_cand = int(
                    any(
                        interval_iou(r["vis_start0"], r["vis_end0"], g["start0"], g["end0"]) >= args.min_iou
                        for g in cand_groups
                    )
                )
                s1_pos = int(
                    any(
                        g["cds_pred"] == 1
                        and interval_iou(r["vis_start0"], r["vis_end0"], g["start0"], g["end0"]) >= args.min_iou
                        for g in cand_groups
                    )
                )

                subsets = ["all_refs"]
                if r["obs_bp"] > 103:
                    subsets.append("observable_bp_gt103")

                for sub in subsets:
                    ta = trunc_acc[(sub, st)]
                    ta["refs"] += 1
                    ta["matched"] += matched
                    ta["exact"] += exact
                    if req_edges:
                        ta["edge_refs"] += 1
                        ta["edge_rec"] += edge_rec

                    da = diag_acc[(sub, st)]
                    da["refs"] += 1
                    da["stage1_cand_covered"] += s1_cand
                    da["stage1_pos_covered"] += s1_pos
                    da["final_matched"] += matched
                    if req_edges:
                        da["edge_refs"] += 1
                        da["edge_rec"] += edge_rec

                ma = mat_acc[(flen, st)]
                ma["refs"] += 1
                ma["matched"] += matched
                ma["exact"] += exact
                if req_edges:
                    ma["edge_refs"] += 1
                    ma["edge_rec"] += edge_rec

        assert tot["tp"] + tot["fn"] == tot["refs"]
        assert tot["tp"] + tot["fp"] == tot["preds"]

        p = safe_div(tot["tp"], tot["preds"])
        r_val = safe_div(tot["tp"], tot["refs"])
        summary_rows.append(
            {
                "mode": mode,
                "fragments": tot["fragments"],
                "total_bp": tot["bp"],
                "refs": tot["refs"],
                "preds": tot["preds"],
                "tp": tot["tp"],
                "fp": tot["fp"],
                "fn": tot["fn"],
                "precision": round(p, 4),
                "recall": round(r_val, 4),
                "f1": round(calc_f1(p, r_val), 4),
                "exact_observable_recall": round(safe_div(tot["exact"], tot["refs"]), 4),
                "edge_refs": tot["edge_refs"],
                "edge_recovered": tot["edge_rec"],
                "partial_edge_recovery": round(safe_div(tot["edge_rec"], tot["edge_refs"]), 4),
                "fp_per_kb": round(safe_div(tot["fp"], tot["bp"] / 1000.0), 4),
                "elapsed_seconds": rt.get("elapsed_seconds", ""),
                "peak_vram_mib": rt.get("peak_vram_mib", ""),
            }
        )

        for sub in ("all_refs", "observable_bp_gt103"):
            for st in STATUSES:
                ta = trunc_acc[(sub, st)]
                trunc_rows.append(
                    {
                        "mode": mode,
                        "subset": sub,
                        "truncation_status": st,
                        "refs": ta["refs"],
                        "matched": ta["matched"],
                        "recall": round(safe_div(ta["matched"], ta["refs"]), 4),
                        "exact_matched": ta["exact"],
                        "exact_recall": round(safe_div(ta["exact"], ta["refs"]), 4),
                        "edge_refs": ta["edge_refs"],
                        "edge_recovered": ta["edge_rec"],
                        "partial_edge_recovery": (
                            round(safe_div(ta["edge_rec"], ta["edge_refs"]), 4)
                            if ta["edge_refs"]
                            else "NA"
                        ),
                    }
                )
                da = diag_acc[(sub, st)]
                diag_rows.append(
                    {
                        "mode": mode,
                        "subset": sub,
                        "truncation_status": st,
                        "refs": da["refs"],
                        "stage1_cand_covered": da["stage1_cand_covered"],
                        "pct_stage1_cand": round(100 * safe_div(da["stage1_cand_covered"], da["refs"]), 2),
                        "stage1_pos_covered": da["stage1_pos_covered"],
                        "pct_stage1_pos": round(100 * safe_div(da["stage1_pos_covered"], da["refs"]), 2),
                        "final_matched": da["final_matched"],
                        "pct_final_matched": round(100 * safe_div(da["final_matched"], da["refs"]), 2),
                        "edge_recovered": da["edge_rec"],
                        "pct_edge_recovered": (
                            round(100 * safe_div(da["edge_rec"], da["edge_refs"]), 2)
                            if da["edge_refs"]
                            else "NA"
                        ),
                    }
                )

        for flen in FRAG_LENGTHS:
            la = len_acc[flen]
            lp = safe_div(la["tp"], la["preds"])
            lr = safe_div(la["tp"], la["refs"])
            frag_len_rows.append(
                {
                    "mode": mode,
                    "length": flen,
                    "fragments": la["fragments"],
                    "refs": la["refs"],
                    "preds": la["preds"],
                    "tp": la["tp"],
                    "fp": la["fp"],
                    "precision": round(lp, 4),
                    "recall": round(lr, 4),
                    "f1": round(calc_f1(lp, lr), 4),
                    "exact_recall": round(safe_div(la["exact"], la["refs"]), 4),
                    "edge_refs": la["edge_refs"],
                    "edge_recovered": la["edge_rec"],
                    "partial_edge_recovery": round(safe_div(la["edge_rec"], la["edge_refs"]), 4),
                    "fp_per_kb": round(safe_div(la["fp"], la["bp"] / 1000.0), 4),
                }
            )
            for st in STATUSES:
                ma = mat_acc[(flen, st)]
                if ma["refs"] == 0:
                    continue
                matrix_rows.append(
                    {
                        "mode": mode,
                        "length": flen,
                        "truncation_status": st,
                        "refs": ma["refs"],
                        "matched": ma["matched"],
                        "recall": round(safe_div(ma["matched"], ma["refs"]), 4),
                        "exact_recall": round(safe_div(ma["exact"], ma["refs"]), 4),
                        "partial_edge_recovery": (
                            round(safe_div(ma["edge_rec"], ma["edge_refs"]), 4)
                            if ma["edge_refs"]
                            else "NA"
                        ),
                    }
                )

    write_tsv(args.out_dir / "resumen_modos_etapa2.tsv", summary_rows, list(summary_rows[0].keys()))
    write_tsv(args.out_dir / "desglose_truncamiento.tsv", trunc_rows, list(trunc_rows[0].keys()))
    write_tsv(
        args.out_dir / "desglose_longitud_fragmento.tsv",
        frag_len_rows,
        list(frag_len_rows[0].keys()),
    )
    write_tsv(
        args.out_dir / "matriz_longitud_x_truncamiento.tsv",
        matrix_rows,
        list(matrix_rows[0].keys()),
    )
    write_tsv(args.out_dir / "diagnostico_parciales.tsv", diag_rows, list(diag_rows[0].keys()))

    b2b_rows = [r for r in read_tsv(args.b2b_path) if r.get("strand") == "ALL"]

    def to_md(rows: list[dict], cols: list[str]) -> str:
        if not rows:
            return "_Sin datos._"
        hdr = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        body = ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
        return "\n".join([hdr, sep, *body])

    md = [
        "# Reporte Experimental — Etapa 2: Comportamiento de GeneLM en Datos Parciales / Truncados",
        "",
        "## 1. Diseño Experimental de Etapa 2",
        "- **Benchmark 2B (Prueba controlada directa en `BacteriaCDS-DNABERT-K6-89M`):** Evalúa la degradación pura del clasificador ante cortes exactos en marco (`complete_gt103`, `sin_5prime_50pct`, `sin_3prime_50pct_slice3p`, `sin_3prime_50pct_slice5p`, `sin_ambos_50pct`, `truncado_corto_le100`) usando las `2,051` CDS reales de los 20 fagos piloto y `2,051` negativos completos muestreados.",
        "- **Benchmark 2A (Pipeline end-to-end sobre fragmentos tipo *contig*):** Evalúa `1,301` fragmentos estratificados (`1,000`: 732, `2,000`: 306, `5,000`: 162, `10,000`: 101; total `3,164,000` pb y `5,767` referencias observables: `3,608 complete`, `1,019 partial_5prime`, `985 partial_3prime`, `155 partial_both`) en 3 modos:",
        "  - `e2_core_original`: pipeline estándar `core.py` (`partial3=False`, `partial5=False`, `coding_len > 100`).",
        "  - `e2_abiertos_gt100`: candidatos en 6 marcos (`complete`, `open_5`, `open_3`, `open_both`) alineados al marco de lectura con `in_frame_len > 100` nt y padding `'N'` en ventanas TIS `< 30` pb.",
        "  - `e2_abiertos_ge60`: igual al anterior pero bajando el corte a `in_frame_len >= 60` nt.",
        "",
        "## 2. Benchmark 2B — Degradación Pura de `BacteriaCDS` ante Truncamiento Controlado",
        to_md(
            b2b_rows,
            [
                "condition",
                "n_pos",
                "n_neg",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "fpr",
                "auroc",
                "mean_prob_pos",
                "mean_prob_neg",
            ],
        ),
        "",
        "## 3. Benchmark 2A — Resultados Globales en Fragmentos (`1, 2, 5, 10 kb`)",
        to_md(
            summary_rows,
            [
                "mode",
                "refs",
                "preds",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
                "exact_observable_recall",
                "partial_edge_recovery",
                "fp_per_kb",
                "elapsed_seconds",
                "peak_vram_mib",
            ],
        ),
        "",
        "## 4. Desglose por Categoría de Truncamiento (Todas vs `observable_bp > 103`)",
        to_md(
            trunc_rows,
            [
                "mode",
                "subset",
                "truncation_status",
                "refs",
                "matched",
                "recall",
                "exact_recall",
                "edge_refs",
                "edge_recovered",
                "partial_edge_recovery",
            ],
        ),
        "",
        "## 5. Desglose por Longitud de Fragmento",
        to_md(
            frag_len_rows,
            [
                "mode",
                "length",
                "fragments",
                "refs",
                "preds",
                "precision",
                "recall",
                "f1",
                "exact_recall",
                "partial_edge_recovery",
                "fp_per_kb",
            ],
        ),
        "",
        "## 6. Diagnóstico por Etapa en Referencias Parciales (`observable_bp > 103`)",
        to_md(
            [r for r in diag_rows if r["subset"] == "observable_bp_gt103"],
            [
                "mode",
                "truncation_status",
                "refs",
                "stage1_cand_covered",
                "pct_stage1_cand",
                "stage1_pos_covered",
                "pct_stage1_pos",
                "final_matched",
                "pct_final_matched",
                "edge_recovered",
                "pct_edge_recovered",
            ],
        ),
    ]
    report_path = args.out_dir / "REPORTE_ETAPA2_DATOS_PARCIALES.md"
    report_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print("Resumen Benchmark 2A:")
    for r in summary_rows:
        print(
            f"  {r['mode']}: P={r['precision']} R={r['recall']} F1={r['f1']} | "
            f"Exact_R={r['exact_observable_recall']} | Partial_Edge_Rec={r['partial_edge_recovery']} | "
            f"FP/kb={r['fp_per_kb']}"
        )
    print(f"Reporte escrito en: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
