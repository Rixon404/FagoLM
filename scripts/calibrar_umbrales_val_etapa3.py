#!/usr/bin/env python3
"""Paso 4 (Etapa 3): Calibra umbrales de decisión de FagoLM-CDS y FagoLM-TIS
exclusivamente sobre los fragmentos de Validación (val_fragmentos/), sin tocar Test.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import time
from collections import defaultdict
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

DEFAULT_STAGE3 = (_REPO_DIR / "etapa3_finetuning_lora"
)
DEFAULT_MODELS = (_PARENT_DIR / "trabajos_referencia/datasets")

STARTS_CORE = ("TTG", "CTG", "ATG", "GTG")
STOPS_CORE = ("TAA", "TAG", "TGA")


def generate_kmer(sequence: str, k: int = 6, overlap: int = 1) -> str:
    if len(sequence) < k:
        return ""
    return " ".join(sequence[j : j + k] for j in range(0, len(sequence) - k + 1, overlap))


def oriented_to_genomic(strand: str, o_start: int, o_end: int, L: int) -> tuple[int, int]:
    if strand == "+":
        return o_start, o_end
    return L - o_end, L - o_start


def extract_tis_window_safe(
    seq_plus: str, seq_rc: str, strand: str, g_start0: int, g_end0: int, w: int = 30
) -> str:
    L = len(seq_plus)
    if strand == "+":
        center = g_start0
        src = seq_plus
    else:
        center = L - g_end0
        src = seq_rc
    left = center - w
    right = center + w
    return ("N" * max(0, -left)) + src[max(0, left) : min(L, right)] + ("N" * max(0, right - L))


def build_open_groups_ge60(frag_id: str, seq_u: str, seq_rc: str) -> list[dict]:
    L = len(seq_u)
    starts_set = set(STARTS_CORE)
    stops_set = set(STOPS_CORE)
    groups: list[dict] = []
    grp_idx = 0

    for strand, oriented in (("+", seq_u), ("-", seq_rc)):
        prefix = "P" if strand == "+" else "N"
        for frame in range(3):
            codon_starts = list(range(frame, L - 2, 3))
            if not codon_starts:
                continue
            codons = [oriented[p : p + 3] for p in codon_starts]
            stop_indices = [i for i, c in enumerate(codons) if c in stops_set]
            start_indices = [i for i, c in enumerate(codons) if c in starts_set]
            suffix_partial = (L - frame) % 3
            aligned_end = L - suffix_partial

            if stop_indices:
                # 1) open_5 (primer stop)
                first_si = stop_indices[0]
                first_stop_end = codon_starts[first_si] + 3
                in_frame_seq = oriented[frame:first_stop_end]
                if len(in_frame_seq) >= 60:
                    grp_idx += 1
                    gs_open5, ge_open5 = oriented_to_genomic(strand, 0, first_stop_end, L)
                    cands = []
                    for si in start_indices:
                        if si >= first_si:
                            break
                        st_pos = codon_starts[si]
                        if codon_starts[first_si] - st_pos >= 10:
                            gs, ge = oriented_to_genomic(strand, st_pos, first_stop_end, L)
                            cands.append({"g_start0": gs, "g_end0": ge})
                    groups.append(
                        {
                            "fragment_id": frag_id,
                            "group_uid": f"{frag_id}::{prefix}{grp_idx}",
                            "strand": strand,
                            "region_kind": "open_5",
                            "span_g_start0": gs_open5,
                            "span_g_end0": ge_open5,
                            "in_frame_len": len(in_frame_seq),
                            "kmer_cds": generate_kmer(in_frame_seq[-510:], k=6, overlap=3),
                            "tis_candidates": cands,
                        }
                    )

                # 2) complete (stops posteriores)
                for s_k in range(1, len(stop_indices)):
                    prev_si = stop_indices[s_k - 1]
                    cur_si = stop_indices[s_k]
                    stop_end = codon_starts[cur_si] + 3
                    cands = []
                    longest_len = 0
                    longest_seq = ""
                    longest_gs, longest_ge = 0, 0
                    for si in start_indices:
                        if prev_si < si < cur_si:
                            st_pos = codon_starts[si]
                            if codon_starts[cur_si] - st_pos >= 10:
                                gs, ge = oriented_to_genomic(strand, st_pos, stop_end, L)
                                tlen = stop_end - st_pos
                                cands.append({"g_start0": gs, "g_end0": ge})
                                if tlen > longest_len:
                                    longest_len = tlen
                                    longest_seq = oriented[st_pos:stop_end]
                                    longest_gs, longest_ge = gs, ge
                    if cands and longest_len >= 60:
                        grp_idx += 1
                        groups.append(
                            {
                                "fragment_id": frag_id,
                                "group_uid": f"{frag_id}::{prefix}{grp_idx}",
                                "strand": strand,
                                "region_kind": "complete",
                                "span_g_start0": longest_gs,
                                "span_g_end0": longest_ge,
                                "in_frame_len": longest_len,
                                "kmer_cds": generate_kmer(longest_seq[-510:], k=6, overlap=3),
                                "tis_candidates": cands,
                            }
                        )

                # 3) open_3 (tras último stop)
                last_si = stop_indices[-1]
                after = [si for si in start_indices if si > last_si]
                if after:
                    cands = []
                    longest_len = 0
                    longest_seq = ""
                    longest_gs, longest_ge = 0, 0
                    for si in after:
                        st_pos = codon_starts[si]
                        if aligned_end - st_pos >= 12:
                            gs, ge = oriented_to_genomic(strand, st_pos, L, L)
                            cands.append({"g_start0": gs, "g_end0": ge})
                            ilen = aligned_end - st_pos
                            if ilen > longest_len:
                                longest_len = ilen
                                longest_seq = oriented[st_pos:aligned_end]
                                longest_gs, longest_ge = gs, ge
                    if cands and longest_len >= 60:
                        grp_idx += 1
                        groups.append(
                            {
                                "fragment_id": frag_id,
                                "group_uid": f"{frag_id}::{prefix}{grp_idx}",
                                "strand": strand,
                                "region_kind": "open_3",
                                "span_g_start0": longest_gs,
                                "span_g_end0": longest_ge,
                                "in_frame_len": longest_len,
                                "kmer_cds": generate_kmer(longest_seq[-510:], k=6, overlap=3),
                                "tis_candidates": cands,
                            }
                        )
            else:
                # 4) open_both
                in_frame_seq = oriented[frame:aligned_end]
                if len(in_frame_seq) >= 60:
                    grp_idx += 1
                    gs_both, ge_both = oriented_to_genomic(strand, 0, L, L)
                    cands = []
                    for si in start_indices:
                        st_pos = codon_starts[si]
                        if aligned_end - st_pos >= 12:
                            gs, ge = oriented_to_genomic(strand, st_pos, L, L)
                            cands.append({"g_start0": gs, "g_end0": ge})
                    groups.append(
                        {
                            "fragment_id": frag_id,
                            "group_uid": f"{frag_id}::{prefix}{grp_idx}",
                            "strand": strand,
                            "region_kind": "open_both",
                            "span_g_start0": gs_both,
                            "span_g_end0": ge_both,
                            "in_frame_len": len(in_frame_seq),
                            "kmer_cds": generate_kmer(in_frame_seq[-510:], k=6, overlap=3),
                            "tis_candidates": cands,
                        }
                    )
    return groups


def predict_probs(model, tokenizer, texts: list[str], device: torch.device, batch_size: int) -> np.ndarray:
    if not texts:
        return np.zeros((0,), dtype=np.float32)
    probs: list[np.ndarray] = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=tokenizer.model_max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            if device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(**inputs).logits.float().cpu().numpy()
            else:
                logits = model(**inputs).logits.float().cpu().numpy()
            shifted = logits - np.max(logits, axis=1, keepdims=True)
            exps = np.exp(shifted)
            p = exps / np.sum(exps, axis=1, keepdims=True)
            probs.append(p[:, 1])
    return np.concatenate(probs, axis=0)


def interval_iou(a0: int, a1: int, b0: int, b1: int) -> float:
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union else 0.0


def maximum_matching_iou(refs: list[dict], preds: list[dict], min_iou: float = 0.5) -> dict[int, int]:
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
    return {ri: pi for pi, ri in pred_to_ref.items()}


def expected_edges(status: str, strand_sym: str) -> set[str]:
    if status == "partial_both":
        return {"left", "right"}
    if status == "partial_5prime":
        return {"left"} if strand_sym == "+" else {"right"}
    if status == "partial_3prime":
        return {"right"} if strand_sym == "+" else {"left"}
    return set()


def evaluate_thresholds(
    windows: list[dict],
    refs_by_frag: dict[str, list[dict]],
    groups_by_frag: dict[str, list[dict]],
    tau_cds_gt100: float,
    tau_cds_short: float,
    tau_tis_default: float,
    tau_tis_open5_override: float,
) -> dict:
    total_bp = 0
    total_refs = 0
    total_preds = 0
    tp = 0
    exact_tp = 0
    edge_refs = 0
    edge_rec = 0

    for w in windows:
        fid = w["fragment_id"]
        flen = int(w["length"])
        total_bp += flen
        frefs = refs_by_frag.get(fid, [])
        fpreds: list[dict] = []

        for g in groups_by_frag.get(fid, []):
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
                fpreds.append({"start0": best["g_start0"], "end0": best["g_end0"], "strand": g["strand"]})
            else:
                best = None
                for c in cands:
                    if c["tis_prob1"] >= tau_tis_open5_override:
                        if best is None or c["tis_prob1"] > best["tis_prob1"]:
                            best = c
                if best is not None:
                    fpreds.append({"start0": best["g_start0"], "end0": best["g_end0"], "strand": g["strand"]})
                else:
                    fpreds.append({"start0": g["span_g_start0"], "end0": g["span_g_end0"], "strand": g["strand"]})

        total_refs += len(frefs)
        total_preds += len(fpreds)
        ref_to_pred = maximum_matching_iou(frefs, fpreds, min_iou=0.5)
        tp += len(ref_to_pred)

        for ri, r in enumerate(frefs):
            req = expected_edges(r["truncation_status"], r["strand_sym"])
            if req:
                edge_refs += 1
            if ri in ref_to_pred:
                p = fpreds[ref_to_pred[ri]]
                if p["start0"] == r["vis_start0"] and p["end0"] == r["vis_end0"]:
                    exact_tp += 1
                if req and ("left" not in req or p["start0"] == 0) and ("right" not in req or p["end0"] == flen):
                    edge_rec += 1

    prec = tp / total_preds if total_preds else 0.0
    rec = tp / total_refs if total_refs else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    fp = total_preds - tp
    return {
        "tau_cds_gt100": tau_cds_gt100,
        "tau_cds_short": tau_cds_short,
        "tau_tis_default": tau_tis_default,
        "tau_tis_open5_override": tau_tis_open5_override,
        "refs": total_refs,
        "preds": total_preds,
        "tp": tp,
        "fp": fp,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "exact_observable_recall": round(exact_tp / total_refs if total_refs else 0.0, 4),
        "partial_edge_recovery": round(edge_rec / edge_refs if edge_refs else 0.0, 4),
        "fp_per_kb": round(fp / (total_bp / 1000.0), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage3-dir", type=Path, default=DEFAULT_STAGE3)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    val_dir = args.stage3_dir / "val_fragmentos"
    out_dir = args.stage3_dir / "calibracion_val"
    out_dir.mkdir(parents=True, exist_ok=True)

    with (val_dir / "ventanas.tsv").open(encoding="utf-8") as fh:
        windows = list(csv.DictReader(fh, delimiter="\t"))
    refs_by_frag: dict[str, list[dict]] = defaultdict(list)
    with (val_dir / "referencias_cds_observables.tsv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            refs_by_frag[r["fragment_id"]].append(
                {
                    "vis_start0": int(r["visible_start_local_0"]),
                    "vis_end0": int(r["visible_end_local_0"]),
                    "strand_sym": "+" if str(r["strand"]) in {"1", "+"} else "-",
                    "truncation_status": r["truncation_status"],
                }
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cds_base_dir = args.models_dir / "genelm_cds"
    tis_base_dir = args.models_dir / "genelm_tis"
    cds_adapter = args.stage3_dir / "modelos" / "fagolm_cds_lora"
    tis_adapter = args.stage3_dir / "modelos" / "fagolm_tis_lora"

    tokenizer = AutoTokenizer.from_pretrained(str(cds_base_dir), local_files_only=True)
    model_cds = PeftModel.from_pretrained(
        AutoModelForSequenceClassification.from_pretrained(str(cds_base_dir), local_files_only=True),
        str(cds_adapter),
    ).merge_and_unload().to(device).eval()
    model_tis = PeftModel.from_pretrained(
        AutoModelForSequenceClassification.from_pretrained(str(tis_base_dir), local_files_only=True),
        str(tis_adapter),
    ).merge_and_unload().to(device).eval()

    t0 = time.perf_counter()
    frag_seqs: dict[str, tuple[str, str]] = {}
    all_groups: list[dict] = []
    groups_by_frag: dict[str, list[dict]] = defaultdict(list)

    for rec in SeqIO.parse(str(val_dir / "fragmentos.fna"), "fasta"):
        seq_u = str(rec.seq).upper()
        seq_rc = str(Seq(seq_u).reverse_complement())
        frag_seqs[rec.id] = (seq_u, seq_rc)
        gs = build_open_groups_ge60(rec.id, seq_u, seq_rc)
        all_groups.extend(gs)
        groups_by_frag[rec.id] = gs

    cds_probs = predict_probs(
        model_cds, tokenizer, [g["kmer_cds"] for g in all_groups], device, args.batch_size
    )
    tis_items: list[tuple[dict, str]] = []
    for g, p1 in zip(all_groups, cds_probs):
        g["cds_prob1"] = float(p1)
        if g["cds_prob1"] >= 0.40:
            s_u, s_rc = frag_seqs[g["fragment_id"]]
            for c in g["tis_candidates"]:
                win = extract_tis_window_safe(s_u, s_rc, g["strand"], c["g_start0"], c["g_end0"])
                tis_items.append((c, generate_kmer(win, k=6, overlap=1)))

    tis_probs = predict_probs(
        model_tis, tokenizer, [x[1] for x in tis_items], device, args.batch_size * 2
    )
    for (c, _), p1 in zip(tis_items, tis_probs):
        c["tis_prob1"] = float(p1)

    inf_dt = time.perf_counter() - t0
    print(f"Inferencia única Val completada en {inf_dt:.2f}s (grupos={len(all_groups)}, tis_cands={len(tis_items)})")

    grid = list(
        itertools.product(
            (0.40, 0.50, 0.60),
            (0.50, 0.65, 0.75, 0.85),
            (0.40, 0.50, 0.60),
            (0.50, 0.70, 0.80, 0.90, 0.95),
        )
    )
    results: list[dict] = []
    for t_gt100, t_short, t_tis, t_open5 in grid:
        res = evaluate_thresholds(
            windows, refs_by_frag, groups_by_frag, t_gt100, t_short, t_tis, t_open5
        )
        results.append(res)

    results_sorted = sorted(
        results,
        key=lambda r: (r["precision"] >= 0.85, r["f1"], r["partial_edge_recovery"], r["precision"]),
        reverse=True,
    )
    best = results_sorted[0]
    default_res = next(
        r
        for r in results
        if r["tau_cds_gt100"] == 0.50
        and r["tau_cds_short"] == 0.50
        and r["tau_tis_default"] == 0.50
        and r["tau_tis_open5_override"] == 0.50
    )

    grid_path = out_dir / "grid_val_results.tsv"
    with grid_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results_sorted[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(results_sorted)

    cal_payload = {
        "best_calibrated": best,
        "default_050": default_res,
        "inference_seconds_val": round(inf_dt, 2),
        "grid_size": len(results),
    }
    (out_dir / "umbrales_calibrados.json").write_text(
        json.dumps(cal_payload, indent=2) + "\n", encoding="utf-8"
    )
    print("Default (0.50) en Val:", json.dumps(default_res))
    print("Mejor calibrado en Val:", json.dumps(best))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
