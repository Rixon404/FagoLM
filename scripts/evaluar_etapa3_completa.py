#!/usr/bin/env python3
"""Paso 5 (Etapa 3): Evaluación integral antes vs. después (FagoLM-LoRA vs GeneLM base)
sobre los 3 conjuntos congelados de Test (20 fagos INPHARED):
  - Test 3A: Benchmark 2B directo en BacteriaCDS vs FagoLM-CDS
  - Test 3B: Benchmark 2A en los 1,301 fragmentos (1/2/5/10 kb; 5,767 CDS observables)
  - Test 3C: Control de no olvido en los 20 genomas completos de Etapa 1 (2,051 CDS)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_REPO_DIR = Path(__file__).resolve().parent.parent
_PARENT_DIR = _REPO_DIR.parent

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import numpy as np
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from calibrar_umbrales_val_etapa3 import (  # noqa: E402
    build_open_groups_ge60,
    expected_edges,
    extract_tis_window_safe,
    generate_kmer,
    maximum_matching_iou,
    oriented_to_genomic,
    predict_probs,
)
from evaluar_truncamiento_directo_cds import (  # noqa: E402
    compute_metrics as compute_b2b_metrics,
    extract_complete_orfs_longest_per_stop,
    parse_pos_cds,
    sample_matched_negatives,
    transform_sequence,
)

DEFAULT_BASE = _REPO_DIR
DEFAULT_MODELS = (_PARENT_DIR / "trabajos_referencia/datasets")

STARTS_CORE = ("TTG", "CTG", "ATG", "GTG")
STOPS_CORE = ("TAA", "TAG", "TGA")


def read_tsv(path: Path) -> list[dict]:
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
    by_seq: dict[str, list[dict]] = defaultdict(list)
    if not path.is_file():
        return by_seq
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw or raw.startswith("#"):
            if raw.strip() == "##FASTA":
                break
            continue
        parts = raw.split("\t")
        if len(parts) != 9:
            continue
        seqid, _src, ftype, s1_s, e1_s, _sc, strand, _ph, _attrs = parts
        if ftype.lower() not in {"cds", "orf"} or strand not in {"+", "-"}:
            continue
        s0, e0 = int(s1_s) - 1, int(e1_s)
        by_seq[seqid].append(
            {
                "start0": s0,
                "end0": e0,
                "strand": strand,
                "stop_coord_0": e0 if strand == "+" else s0,
            }
        )
    return by_seq


def run_test_3a_b2b(
    base_dir: Path, model_cds, tokenizer, device: torch.device, batch_size: int
) -> list[dict]:
    piloto_dir = base_dir / "etapa1_fagos_completos" / "piloto_inphared"
    with (piloto_dir / "manifiesto_piloto.tsv").open(encoding="utf-8") as fh:
        manifest = list(csv.DictReader(fh, delimiter="\t"))

    pos_all: list[dict] = []
    neg_pool: list[dict] = []
    for row in manifest:
        acc = row["accession"]
        rec = next(SeqIO.parse(str(piloto_dir / "genomas" / f"{acc}.fna"), "fasta"))
        seq_u = str(rec.seq).upper()
        pos_rows = parse_pos_cds(piloto_dir / "genomas" / f"{acc}.gff", seq_u, rec.id)
        pos_stop_set = {(r["strand"], r["stop_coord_0"]) for r in pos_rows}
        pos_all.extend(pos_rows)
        for orf in extract_complete_orfs_longest_per_stop(seq_u):
            if (orf["strand"], orf["stop_coord_0"]) not in pos_stop_set:
                neg_pool.append(
                    {
                        "record_id": rec.id,
                        "strand": orf["strand"],
                        "stop_coord_0": orf["stop_coord_0"],
                        "sequence": orf["sequence"],
                        "label": 0,
                    }
                )
    neg_sampled = sample_matched_negatives(pos_all, neg_pool, seed=42)
    dataset = pos_all + neg_sampled

    # Cargar baseline Etapa 2 (Benchmark 2B)
    b2b_base_path = base_dir / "etapa2_datos_parciales" / "benchmark_directo_cds" / "resultados_truncamiento_directo.tsv"
    base_by_key = {(r["condition"], r["strand"]): r for r in read_tsv(b2b_base_path)}

    conditions = (
        "complete_gt103",
        "sin_5prime_50pct",
        "sin_3prime_50pct_slice3p",
        "sin_3prime_50pct_slice5p",
        "sin_ambos_50pct",
        "truncado_corto_le100",
    )

    comp_rows: list[dict] = []
    for cond in conditions:
        items = []
        kmers = []
        for r in dataset:
            sub = transform_sequence(r["sequence"], cond)
            if sub is None:
                continue
            items.append((r["label"], r["strand"]))
            kmers.append(generate_kmer(sub, k=6, overlap=3))

        p1 = predict_probs(model_cds, tokenizer, kmers, device, batch_size)
        probs_2d = np.stack([1.0 - p1, p1], axis=1)
        y_true = np.array([it[0] for it in items], dtype=int)
        strands = np.array([it[1] for it in items])

        for strand_lbl, mask in (
            ("ALL", np.ones(len(y_true), dtype=bool)),
            ("+", strands == "+"),
            ("-", strands == "-"),
        ):
            m_after = compute_b2b_metrics(y_true[mask], probs_2d[mask])
            b_row = base_by_key[(cond, strand_lbl)]
            comp_rows.append(
                {
                    "condition": cond,
                    "strand": strand_lbl,
                    "n_pos": m_after["n_pos"],
                    "n_neg": m_after["n_neg"],
                    "base_recall": float(b_row["recall"]),
                    "lora_recall": m_after["recall"],
                    "delta_recall": round(m_after["recall"] - float(b_row["recall"]), 4),
                    "base_precision": float(b_row["precision"]),
                    "lora_precision": m_after["precision"],
                    "base_f1": float(b_row["f1"]),
                    "lora_f1": m_after["f1"],
                    "delta_f1": round(m_after["f1"] - float(b_row["f1"]), 4),
                    "base_fpr": float(b_row["fpr"]),
                    "lora_fpr": m_after["fpr"],
                    "delta_fpr": round(m_after["fpr"] - float(b_row["fpr"]), 4),
                    "base_auroc": float(b_row["auroc"]),
                    "lora_auroc": m_after["auroc"],
                    "base_mean_prob_pos": float(b_row["mean_prob_pos"]),
                    "lora_mean_prob_pos": m_after["mean_prob_pos"],
                    "base_mean_prob_neg": float(b_row["mean_prob_neg"]),
                    "lora_mean_prob_neg": m_after["mean_prob_neg"],
                }
            )
    return comp_rows


def decode_predictions(
    groups_by_frag: dict[str, list[dict]],
    windows: list[dict],
    mode_name: str,
    min_len_gt100: bool,
    tau_cds_gt100: float,
    tau_cds_short: float,
    tau_tis_default: float,
    tau_tis_open5_override: float,
    out_gff: Path,
) -> dict[str, list[dict]]:
    preds_by_frag: dict[str, list[dict]] = defaultdict(list)
    gff_lines = ["##gff-version 3\n", f"##Tool: FagoLM-LoRA ({mode_name})\n"]
    idx = 0
    for w in windows:
        fid = w["fragment_id"]
        for g in groups_by_frag.get(fid, []):
            if min_len_gt100:
                if g["region_kind"] in ("complete", "open_5"):
                    if g["in_frame_len"] <= 103:
                        continue
                else:
                    if g["in_frame_len"] <= 100:
                        continue
            tau_cds = tau_cds_short if g["in_frame_len"] <= 103 else tau_cds_gt100
            if g["cds_prob1"] < tau_cds:
                continue

            rk = g["region_kind"]
            cands = g["tis_candidates"]
            if rk in ("complete", "open_3"):
                best = None
                for c in cands:
                    if c["tis_prob1"] >= tau_tis_default:
                        if best is None or c["tis_prob1"] > best["tis_prob1"]:
                            best = c
                if best is None:
                    continue
                gs, ge = best["g_start0"], best["g_end0"]
                score = best["tis_prob1"]
                cand_type = rk
                partial_flag = "00" if rk == "complete" else "01"
            else:
                best = None
                for c in cands:
                    if c["tis_prob1"] >= tau_tis_open5_override:
                        if best is None or c["tis_prob1"] > best["tis_prob1"]:
                            best = c
                if best is not None:
                    gs, ge = best["g_start0"], best["g_end0"]
                    score = best["tis_prob1"]
                    cand_type = "complete" if rk == "open_5" else "open_3"
                    partial_flag = "00" if rk == "open_5" else "01"
                else:
                    gs, ge = g["span_g_start0"], g["span_g_end0"]
                    score = g["cds_prob1"]
                    cand_type = rk
                    partial_flag = "10" if rk == "open_5" else "11"

            preds_by_frag[fid].append({"start0": gs, "end0": ge, "strand": g["strand"], "cand_type": cand_type})
            idx += 1
            attrs = f"ID={mode_name}_{idx:06d};cand_type={cand_type};partial={partial_flag};cds_prob1={g['cds_prob1']:.4f}"
            gff_lines.append(
                "\t".join([fid, f"FagoLM_{mode_name}", "CDS", str(gs + 1), str(ge), f"{score:.4f}", g["strand"], "0", attrs]) + "\n"
            )
    out_gff.parent.mkdir(parents=True, exist_ok=True)
    out_gff.write_text("".join(gff_lines), encoding="utf-8")
    return preds_by_frag


def score_fragment_predictions(
    mode: str,
    windows: list[dict],
    refs_by_frag: dict[str, list[dict]],
    preds_by_frag: dict[str, list[dict]],
    elapsed_s: float,
    peak_vram_mib: float,
) -> tuple[dict, list[dict], list[dict]]:
    tot = Counter()
    len_acc = defaultdict(Counter)
    trunc_acc = defaultdict(Counter)

    for w in windows:
        fid = w["fragment_id"]
        flen = int(w["length"])
        frefs = refs_by_frag.get(fid, [])
        fpreds = preds_by_frag.get(fid, [])
        ref_to_pred = maximum_matching_iou(frefs, fpreds, min_iou=0.5)

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
            exact = int(matched and pred["start0"] == r["vis_start0"] and pred["end0"] == r["vis_end0"])
            req = expected_edges(st, r["strand_sym"])
            edge_rec = int(
                matched
                and bool(req)
                and ("left" not in req or pred["start0"] == 0)
                and ("right" not in req or pred["end0"] == flen)
            )
            tot["exact"] += exact
            len_acc[flen]["exact"] += exact
            if req:
                tot["edge_refs"] += 1
                tot["edge_rec"] += edge_rec
                len_acc[flen]["edge_refs"] += 1
                len_acc[flen]["edge_rec"] += edge_rec

            subsets = ["all_refs", "observable_bp_gt103" if r["obs_bp"] > 103 else "observable_bp_le103"]
            for sub in subsets:
                ta = trunc_acc[(sub, st)]
                ta["refs"] += 1
                ta["matched"] += matched
                ta["exact"] += exact
                if req:
                    ta["edge_refs"] += 1
                    ta["edge_rec"] += edge_rec

    assert tot["tp"] + tot["fn"] == tot["refs"]
    assert tot["tp"] + tot["fp"] == tot["preds"]

    p = safe_div(tot["tp"], tot["preds"])
    r_val = safe_div(tot["tp"], tot["refs"])
    global_row = {
        "mode": mode,
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
        "elapsed_seconds": round(elapsed_s, 2),
        "peak_vram_mib": round(peak_vram_mib, 2),
    }

    trunc_rows = []
    for sub in ("all_refs", "observable_bp_gt103", "observable_bp_le103"):
        for st in ("complete", "partial_5prime", "partial_3prime", "partial_both"):
            ta = trunc_acc[(sub, st)]
            if ta["refs"] == 0:
                continue
            trunc_rows.append(
                {
                    "mode": mode,
                    "subset": sub,
                    "truncation_status": st,
                    "refs": ta["refs"],
                    "matched": ta["matched"],
                    "recall": round(safe_div(ta["matched"], ta["refs"]), 4),
                    "exact_recall": round(safe_div(ta["exact"], ta["refs"]), 4),
                    "edge_refs": ta["edge_refs"],
                    "edge_recovered": ta["edge_rec"],
                    "partial_edge_recovery": round(safe_div(ta["edge_rec"], ta["edge_refs"]), 4) if ta["edge_refs"] else "NA",
                }
            )

    len_rows = []
    for flen in (1000, 2000, 5000, 10000):
        la = len_acc[flen]
        lp = safe_div(la["tp"], la["preds"])
        lr = safe_div(la["tp"], la["refs"])
        len_rows.append(
            {
                "mode": mode,
                "length": flen,
                "fragments": la["fragments"],
                "refs": la["refs"],
                "preds": la["preds"],
                "precision": round(lp, 4),
                "recall": round(lr, 4),
                "f1": round(calc_f1(lp, lr), 4),
                "exact_recall": round(safe_div(la["exact"], la["refs"]), 4),
                "partial_edge_recovery": round(safe_div(la["edge_rec"], la["edge_refs"]), 4),
                "fp_per_kb": round(safe_div(la["fp"], la["bp"] / 1000.0), 4),
            }
        )
    return global_row, trunc_rows, len_rows


def build_complete_genome_groups(seq_u: str, seq_rc: str) -> list[dict]:
    """Extrae grupos completos (starts=['TTG','CTG','ATG','GTG'], total_len >= 60)."""
    L = len(seq_u)
    starts_set = set(STARTS_CORE)
    stops_set = set(STOPS_CORE)
    groups: list[dict] = []
    for strand, oriented in (("+", seq_u), ("-", seq_rc)):
        for frame in range(3):
            cur_starts: list[int] = []
            for pos in range(frame, L - 2, 3):
                cod = oriented[pos : pos + 3]
                if cod in stops_set:
                    end_local = pos + 3
                    cands = []
                    for st_pos in cur_starts:
                        if pos - st_pos >= 10:
                            gs, ge = oriented_to_genomic(strand, st_pos, end_local, L)
                            cands.append({"g_start0": gs, "g_end0": ge, "total_len": end_local - st_pos})
                    if cands:
                        first_st = cur_starts[0]
                        total_len = end_local - first_st
                        if total_len >= 60:
                            orf_seq = oriented[first_st:end_local]
                            gs, ge = oriented_to_genomic(strand, first_st, end_local, L)
                            groups.append(
                                {
                                    "strand": strand,
                                    "in_frame_len": total_len,
                                    "coding_len": total_len - 3,
                                    "kmer_cds": generate_kmer(orf_seq[-510:], k=6, overlap=3),
                                    "tis_candidates": cands,
                                    "span_g_start0": gs,
                                    "span_g_end0": ge,
                                }
                            )
                    cur_starts = []
                elif cod in starts_set:
                    cur_starts.append(pos)
    return groups


def score_complete_genomes(mode: str, genomes_info: list[dict], preds_by_acc: dict[str, list[dict]]) -> dict:
    tot = Counter()
    for g in genomes_info:
        acc = g["accession"]
        refs = g["refs"]
        preds = preds_by_acc.get(acc, [])
        ref_to_pred_iou = maximum_matching_iou(refs, preds, min_iou=0.5)

        # Match por stop
        refs_by_sk = defaultdict(list)
        preds_by_sk = defaultdict(list)
        for ri, r in enumerate(refs):
            refs_by_sk[(r["strand_sym"], r["stop_coord_0"])].append(ri)
        for pi, p in enumerate(preds):
            preds_by_sk[(p["strand"], p["stop_coord_0"])].append(pi)
        stop_tp = 0
        exact_tp = 0
        exact_le100 = 0
        exact_101_300 = 0
        refs_le100 = 0
        refs_101_300 = 0
        stop_le100 = 0
        stop_101_300 = 0
        for r in refs:
            if r["len_bp"] <= 100:
                refs_le100 += 1
            elif r["len_bp"] <= 300:
                refs_101_300 += 1

        for sk, r_idxs in refs_by_sk.items():
            p_idxs = preds_by_sk.get(sk, [])
            if p_idxs:
                stop_tp += 1
                r0 = refs[r_idxs[0]]
                p0 = preds[p_idxs[0]]
                if r0["len_bp"] <= 100:
                    stop_le100 += 1
                elif r0["len_bp"] <= 300:
                    stop_101_300 += 1
                if r0["vis_start0"] == p0["start0"] and r0["vis_end0"] == p0["end0"]:
                    exact_tp += 1
                    if r0["len_bp"] <= 100:
                        exact_le100 += 1
                    elif r0["len_bp"] <= 300:
                        exact_101_300 += 1

        tot["bp"] += g["length_bp"]
        tot["refs"] += len(refs)
        tot["preds"] += len(preds)
        tot["tp_iou"] += len(ref_to_pred_iou)
        tot["stop_tp"] += stop_tp
        tot["exact_tp"] += exact_tp
        tot["refs_le100"] += refs_le100
        tot["refs_101_300"] += refs_101_300
        tot["stop_le100"] += stop_le100
        tot["stop_101_300"] += stop_101_300
        tot["exact_le100"] += exact_le100
        tot["exact_101_300"] += exact_101_300

    p_iou = safe_div(tot["tp_iou"], tot["preds"])
    r_iou = safe_div(tot["tp_iou"], tot["refs"])
    sp = safe_div(tot["stop_tp"], tot["preds"])
    sr = safe_div(tot["stop_tp"], tot["refs"])
    return {
        "mode": mode,
        "refs": tot["refs"],
        "preds": tot["preds"],
        "precision_iou": round(p_iou, 4),
        "recall_iou": round(r_iou, 4),
        "f1_iou": round(calc_f1(p_iou, r_iou), 4),
        "stop_precision": round(sp, 4),
        "stop_recall": round(sr, 4),
        "stop_f1": round(calc_f1(sp, sr), 4),
        "exact_recall": round(safe_div(tot["exact_tp"], tot["refs"]), 4),
        "tis_acc_cond": round(safe_div(tot["exact_tp"], tot["stop_tp"]), 4),
        "stop_recall_le100": round(safe_div(tot["stop_le100"], tot["refs_le100"]), 4),
        "exact_recall_le100": round(safe_div(tot["exact_le100"], tot["refs_le100"]), 4),
        "stop_recall_101_300": round(safe_div(tot["stop_101_300"], tot["refs_101_300"]), 4),
        "exact_recall_101_300": round(safe_div(tot["exact_101_300"], tot["refs_101_300"]), 4),
        "fp_per_kb": round(safe_div(tot["preds"] - tot["tp_iou"], tot["bp"] / 1000.0), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    stage3_dir = args.base_dir / "etapa3_finetuning_lora"
    res_dir = stage3_dir / "resultados"
    res_dir.mkdir(parents=True, exist_ok=True)

    cal_path = stage3_dir / "calibracion_val" / "umbrales_calibrados.json"
    cal = json.loads(cal_path.read_text(encoding="utf-8"))["best_calibrated"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cds_base_dir = args.models_dir / "genelm_cds"
    tis_base_dir = args.models_dir / "genelm_tis"
    tokenizer = AutoTokenizer.from_pretrained(str(cds_base_dir), local_files_only=True)
    model_cds = PeftModel.from_pretrained(
        AutoModelForSequenceClassification.from_pretrained(str(cds_base_dir), local_files_only=True),
        str(stage3_dir / "modelos" / "fagolm_cds_lora"),
    ).merge_and_unload().to(device).eval()
    model_tis = PeftModel.from_pretrained(
        AutoModelForSequenceClassification.from_pretrained(str(tis_base_dir), local_files_only=True),
        str(stage3_dir / "modelos" / "fagolm_tis_lora"),
    ).merge_and_unload().to(device).eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # --- TEST 3A: Benchmark 2B directo ---
    print("Ejecutando Test 3A (Benchmark 2B directo)...")
    b2b_comp = run_test_3a_b2b(args.base_dir, model_cds, tokenizer, device, args.batch_size)
    write_tsv(res_dir / "comparacion_benchmark2b_antes_despues.tsv", b2b_comp, list(b2b_comp[0].keys()))
    for r in b2b_comp:
        if r["strand"] == "ALL":
            print(
                f"  [2B {r['condition']:24s}] Rec: {r['base_recall']} -> {r['lora_recall']} ({r['delta_recall']:+.4f}) | "
                f"F1: {r['base_f1']} -> {r['lora_f1']} | FPR: {r['base_fpr']} -> {r['lora_fpr']} | AUROC: {r['base_auroc']} -> {r['lora_auroc']}"
            )

    # --- TEST 3B: Benchmark 2A en 1,301 fragmentos de Test ---
    print("Ejecutando Test 3B (1,301 fragmentos congelados de Test)...")
    frag_dir = args.base_dir / "etapa2_datos_parciales" / "dataset_fragmentos"
    windows = read_tsv(frag_dir / "ventanas.tsv")
    refs_by_frag: dict[str, list[dict]] = defaultdict(list)
    for r in read_tsv(frag_dir / "referencias_cds_observables.tsv"):
        refs_by_frag[r["fragment_id"]].append(
            {
                "vis_start0": int(r["visible_start_local_0"]),
                "vis_end0": int(r["visible_end_local_0"]),
                "obs_bp": int(r["observable_bp"]),
                "strand_sym": "+" if str(r["strand"]) in {"1", "+"} else "-",
                "truncation_status": r["truncation_status"],
            }
        )

    t0 = time.perf_counter()
    frag_seqs: dict[str, tuple[str, str]] = {}
    all_groups: list[dict] = []
    groups_by_frag: dict[str, list[dict]] = defaultdict(list)
    for rec in SeqIO.parse(str(frag_dir / "fragmentos.fna"), "fasta"):
        seq_u = str(rec.seq).upper()
        seq_rc = str(Seq(seq_u).reverse_complement())
        frag_seqs[rec.id] = (seq_u, seq_rc)
        gs = build_open_groups_ge60(rec.id, seq_u, seq_rc)
        all_groups.extend(gs)
        groups_by_frag[rec.id] = gs

    cds_probs = predict_probs(model_cds, tokenizer, [g["kmer_cds"] for g in all_groups], device, args.batch_size)
    min_tau_cds = min(0.50, cal["tau_cds_gt100"], cal["tau_cds_short"])
    tis_items: list[tuple[dict, str]] = []
    for g, p1 in zip(all_groups, cds_probs):
        g["cds_prob1"] = float(p1)
        if g["cds_prob1"] >= min_tau_cds:
            s_u, s_rc = frag_seqs[g["fragment_id"]]
            for c in g["tis_candidates"]:
                win = extract_tis_window_safe(s_u, s_rc, g["strand"], c["g_start0"], c["g_end0"])
                tis_items.append((c, generate_kmer(win, k=6, overlap=1)))

    tis_probs = predict_probs(model_tis, tokenizer, [x[1] for x in tis_items], device, args.batch_size * 2)
    for (c, _), p1 in zip(tis_items, tis_probs):
        c["tis_prob1"] = float(p1)
    inf_frag_s = time.perf_counter() - t0
    peak_mib = round(torch.cuda.max_memory_allocated(device) / (1024**2), 2) if device.type == "cuda" else 0.0

    # Evaluar los 3 modos Etapa 2 previos + 3 modos Etapa 3 con la misma función exacta
    global_comp: list[dict] = []
    trunc_comp: list[dict] = []
    len_comp: list[dict] = []

    e2_rt_dir = args.base_dir / "etapa2_datos_parciales" / "predicciones"
    for e2_mode in ("e2_core_original", "e2_abiertos_gt100", "e2_abiertos_ge60"):
        preds = parse_gff_preds(e2_rt_dir / e2_mode / "predicciones.gff")
        rt = json.loads((e2_rt_dir / e2_mode / "runtime_stats.json").read_text(encoding="utf-8"))
        g_row, t_rows, l_rows = score_fragment_predictions(
            e2_mode, windows, refs_by_frag, preds, rt.get("elapsed_seconds", 0.0), rt.get("peak_vram_mib", 0.0)
        )
        global_comp.append(g_row)
        trunc_comp.extend(t_rows)
        len_comp.extend(l_rows)

    e3_configs = (
        ("e3_fagolm_gt100_t05", True, 0.50, 0.50, 0.50, 0.50),
        ("e3_fagolm_ge60_t05", False, 0.50, 0.50, 0.50, 0.50),
        (
            "e3_fagolm_ge60_cal",
            False,
            cal["tau_cds_gt100"],
            cal["tau_cds_short"],
            cal["tau_tis_default"],
            cal["tau_tis_open5_override"],
        ),
    )
    for m_name, gt100_only, t_gt100, t_short, t_tis, t_open5 in e3_configs:
        preds = decode_predictions(
            groups_by_frag,
            windows,
            m_name,
            gt100_only,
            t_gt100,
            t_short,
            t_tis,
            t_open5,
            stage3_dir / "predicciones" / m_name / "predicciones.gff",
        )
        g_row, t_rows, l_rows = score_fragment_predictions(
            m_name, windows, refs_by_frag, preds, inf_frag_s, peak_mib
        )
        global_comp.append(g_row)
        trunc_comp.extend(t_rows)
        len_comp.extend(l_rows)

    write_tsv(res_dir / "comparacion_etapa2_vs_etapa3_global.tsv", global_comp, list(global_comp[0].keys()))
    write_tsv(res_dir / "comparacion_etapa2_vs_etapa3_truncamiento.tsv", trunc_comp, list(trunc_comp[0].keys()))
    write_tsv(res_dir / "comparacion_etapa2_vs_etapa3_longitud.tsv", len_comp, list(len_comp[0].keys()))

    print("Resumen comparativo Test 3B (Etapa 2 vs Etapa 3):")
    for r in global_comp:
        print(
            f"  {r['mode']:20s}: P={r['precision']} R={r['recall']} F1={r['f1']} | "
            f"Exact_R={r['exact_observable_recall']} PartialEdge={r['partial_edge_recovery']} FP/kb={r['fp_per_kb']}"
        )

    # --- TEST 3C: Control de no olvido en los 20 genomas completos de Etapa 1 ---
    print("Ejecutando Test 3C (20 genomas completos de Etapa 1)...")
    piloto_dir = args.base_dir / "etapa1_fagos_completos" / "piloto_inphared"
    manifest = read_tsv(piloto_dir / "manifiesto_piloto.tsv")
    genomes_info: list[dict] = []
    comp_groups_by_acc: dict[str, list[dict]] = {}
    all_comp_groups: list[dict] = []
    comp_seqs: dict[str, tuple[str, str]] = {}

    for mrow in manifest:
        acc = mrow["accession"]
        rec = next(SeqIO.parse(str(piloto_dir / "genomas" / f"{acc}.fna"), "fasta"))
        seq_u = str(rec.seq).upper()
        seq_rc = str(Seq(seq_u).reverse_complement())
        comp_seqs[acc] = (seq_u, seq_rc)

        refs = []
        for raw in (piloto_dir / "genomas" / f"{acc}.gff").read_text(encoding="utf-8").splitlines():
            if not raw or raw.startswith("#"):
                continue
            p = raw.split("\t")
            if len(p) != 9 or p[2].lower() != "cds" or p[6] not in {"+", "-"}:
                continue
            s0, e0 = int(p[3]) - 1, int(p[4])
            refs.append(
                {
                    "vis_start0": s0,
                    "vis_end0": e0,
                    "len_bp": e0 - s0,
                    "strand_sym": p[6],
                    "stop_coord_0": e0 if p[6] == "+" else s0,
                }
            )
        genomes_info.append({"accession": acc, "length_bp": len(seq_u), "refs": refs})
        gs = build_complete_genome_groups(seq_u, seq_rc)
        for g in gs:
            g["accession"] = acc
        comp_groups_by_acc[acc] = gs
        all_comp_groups.extend(gs)

    comp_cds_p = predict_probs(model_cds, tokenizer, [g["kmer_cds"] for g in all_comp_groups], device, args.batch_size)
    comp_tis_items: list[tuple[dict, str]] = []
    min_comp_cds = min(0.50, cal["tau_cds_gt100"], cal["tau_cds_short"])
    for g, p1 in zip(all_comp_groups, comp_cds_p):
        g["cds_prob1"] = float(p1)
        if g["cds_prob1"] >= min_comp_cds:
            s_u, s_rc = comp_seqs[g["accession"]]
            for c in g["tis_candidates"]:
                win = extract_tis_window_safe(s_u, s_rc, g["strand"], c["g_start0"], c["g_end0"])
                comp_tis_items.append((c, generate_kmer(win, k=6, overlap=1)))
    comp_tis_p = predict_probs(model_tis, tokenizer, [x[1] for x in comp_tis_items], device, args.batch_size * 2)
    for (c, _), p1 in zip(comp_tis_items, comp_tis_p):
        c["tis_prob1"] = float(p1)

    comp_table: list[dict] = []
    e1_pred_dir = args.base_dir / "etapa1_fagos_completos" / "predicciones"
    for e1_mode in ("modo_a_original", "modo_b_adaptado_fagos"):
        p_map = {}
        for mrow in manifest:
            acc = mrow["accession"]
            parsed = parse_gff_preds(e1_pred_dir / e1_mode / f"{acc}.gff")
            p_map[acc] = parsed.get(acc, [])
        comp_table.append(score_complete_genomes(e1_mode, genomes_info, p_map))

    for c_mode, t_gt100, t_short, t_tis in (
        ("e3_fagolm_completos_ge60_t05", 0.50, 0.50, 0.50),
        ("e3_fagolm_completos_ge60_cal", cal["tau_cds_gt100"], cal["tau_cds_short"], cal["tau_tis_default"]),
    ):
        p_map: dict[str, list[dict]] = {}
        out_c_dir = stage3_dir / "predicciones" / c_mode
        out_c_dir.mkdir(parents=True, exist_ok=True)
        for g_info in genomes_info:
            acc = g_info["accession"]
            calls = []
            gff_lines = ["##gff-version 3\n"]
            idx = 0
            for g in comp_groups_by_acc[acc]:
                tau_c = t_short if g["in_frame_len"] <= 103 else t_gt100
                if g["cds_prob1"] < tau_c:
                    continue
                best = None
                for c in g["tis_candidates"]:
                    if c["tis_prob1"] >= t_tis:
                        if best is None or c["tis_prob1"] > best["tis_prob1"]:
                            best = c
                if best is None:
                    continue
                gs, ge = best["g_start0"], best["g_end0"]
                calls.append(
                    {
                        "start0": gs,
                        "end0": ge,
                        "strand": g["strand"],
                        "stop_coord_0": ge if g["strand"] == "+" else gs,
                    }
                )
                idx += 1
                gff_lines.append(
                    "\t".join([acc, f"FagoLM_{c_mode}", "CDS", str(gs + 1), str(ge), f"{best['tis_prob1']:.4f}", g["strand"], "0", f"ID={acc}_{idx:05d}"]) + "\n"
                )
            (out_c_dir / f"{acc}.gff").write_text("".join(gff_lines), encoding="utf-8")
            p_map[acc] = calls
        comp_table.append(score_complete_genomes(c_mode, genomes_info, p_map))

    write_tsv(res_dir / "comparacion_etapa1_completos_antes_despues.tsv", comp_table, list(comp_table[0].keys()))
    print("Resumen comparativo Test 3C (Genomas completos):")
    for r in comp_table:
        print(
            f"  {r['mode']:29s}: IoU_F1={r['f1_iou']} Stop_F1={r['stop_f1']} Exact_R={r['exact_recall']} "
            f"TIS_cond={r['tis_acc_cond']} <=100_R={r['stop_recall_le100']} 101_300_R={r['stop_recall_101_300']}"
        )

    # --- Generar reporte maestro Markdown ---
    split_sum = json.loads((stage3_dir / "resumen_split_etapa3.json").read_text(encoding="utf-8"))
    cds_tr_m = json.loads((stage3_dir / "modelos" / "fagolm_cds_lora" / "training_metrics.json").read_text(encoding="utf-8"))
    tis_tr_m = json.loads((stage3_dir / "modelos" / "fagolm_tis_lora" / "training_metrics.json").read_text(encoding="utf-8"))

    def to_md(rows: list[dict], cols: list[str]) -> str:
        hdr = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        body = ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
        return "\n".join([hdr, sep, *body])

    md = [
        "# Reporte Experimental — Etapa 3: Transfer Learning / Fine-Tuning con LoRA (`FagoLM-CDS` + `FagoLM-TIS`)",
        "",
        "## 1. Split sin Fuga y Entrenamiento LoRA",
        f"- **Particiones estrictamente disjuntas por genoma y homología (21-mers < 0.05):** Train = `{split_sum['train_genomes']}` fagos (`max_cont(Train,Test)={split_sum['max_containment_train_test']}`), Val = `{split_sum['val_genomes']}` fagos (`max_cont(Val,Test)={split_sum['max_containment_val_test']}`, `max_cont(Val,Train)={split_sum['max_containment_val_train']}`), Test = `{split_sum['test_genomes']}` fagos congelados (Etapas 1 y 2). Excluidos por homología: `{split_sum['excluded_test_homology']}` vs Test, `{split_sum['excluded_train_homology']}` vs Train.",
        f"- **`FagoLM-CDS` (LoRA `r=16, alpha=32` sobre `genelm_cds`):** `{cds_tr_m['train_rows']:,}` ejemplos train / `{cds_tr_m['val_rows']:,}` val; `{cds_tr_m['elapsed_seconds']}s`; pico VRAM `{cds_tr_m['peak_vram_mib']} MiB`; Val F1 = `{cds_tr_m['val_overall']['f1']}`, Val AUROC = `{cds_tr_m['val_overall']['auroc']}` (en `len<=103` nt: F1 = `{cds_tr_m['val_breakdown'].get('len_le_103', {}).get('f1')}`, FPR = `{cds_tr_m['val_breakdown'].get('len_le_103', {}).get('fpr')}`, AUROC = `{cds_tr_m['val_breakdown'].get('len_le_103', {}).get('auroc')}`).",
        f"- **`FagoLM-TIS` (LoRA `r=16, alpha=32` sobre `genelm_tis`):** `{tis_tr_m['train_rows']:,}` ejemplos train / `{tis_tr_m['val_rows']:,}` val (50% negativos duros internos en bordes 5' abiertos); `{tis_tr_m['elapsed_seconds']}s`; pico VRAM `{tis_tr_m['peak_vram_mib']} MiB`; Val F1 = `{tis_tr_m['val_overall']['f1']}`, Val AUROC = `{tis_tr_m['val_overall']['auroc']}`.",
        f"- **Umbrales calibrados únicamente en Val:** `tau_cds_gt100={cal['tau_cds_gt100']}`, `tau_cds_short={cal['tau_cds_short']}`, `tau_tis_default={cal['tau_tis_default']}`, `tau_tis_open5_override={cal['tau_tis_open5_override']}` (Val F1 = `{cal['f1']}`, Precision = `{cal['precision']}`, Recall = `{cal['recall']}`, Partial Edge = `{cal['partial_edge_recovery']}`).",
        "",
        "## 2. Test 3A — Benchmark 2B Directo (`genelm_cds` Base vs `FagoLM-CDS` LoRA)",
        to_md(
            [r for r in b2b_comp if r["strand"] == "ALL"],
            [
                "condition", "n_pos", "n_neg", "base_recall", "lora_recall", "delta_recall",
                "base_f1", "lora_f1", "delta_f1", "base_fpr", "lora_fpr", "base_auroc", "lora_auroc",
            ],
        ),
        "",
        "## 3. Test 3B — Benchmark 2A End-to-End en `1,301` Fragmentos (`1, 2, 5, 10 kb`; `5,767` CDS)",
        to_md(
            global_comp,
            [
                "mode", "refs", "preds", "precision", "recall", "f1",
                "exact_observable_recall", "partial_edge_recovery", "fp_per_kb",
            ],
        ),
        "",
        "### 3.1 Desglose por Categoría de Truncamiento (Subconjunto `all_refs`, `observable_bp_gt103` y `observable_bp_le103`)",
        to_md(
            [r for r in trunc_comp if r["mode"] in ("e2_core_original", "e2_abiertos_gt100", "e2_abiertos_ge60", "e3_fagolm_gt100_t05", "e3_fagolm_ge60_t05", "e3_fagolm_ge60_cal")],
            [
                "mode", "subset", "truncation_status", "refs", "matched", "recall",
                "exact_recall", "partial_edge_recovery",
            ],
        ),
        "",
        "### 3.2 Desglose por Longitud de Fragmento",
        to_md(
            len_comp,
            [
                "mode", "length", "refs", "preds", "precision", "recall", "f1",
                "exact_recall", "partial_edge_recovery", "fp_per_kb",
            ],
        ),
        "",
        "## 4. Test 3C — Control de No Olvido en `20` Genomas Completos de Etapa 1 (`2,051` CDS)",
        to_md(
            comp_table,
            [
                "mode", "refs", "preds", "precision_iou", "recall_iou", "f1_iou",
                "stop_f1", "exact_recall", "tis_acc_cond", "stop_recall_le100", "stop_recall_101_300", "fp_per_kb",
            ],
        ),
    ]
    report_path = res_dir / "REPORTE_ETAPA3_FINETUNING_LORA.md"
    report_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Reporte maestro Etapa 3 escrito en: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
