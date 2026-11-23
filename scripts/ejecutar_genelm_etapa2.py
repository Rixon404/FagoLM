#!/usr/bin/env python3
"""Benchmark 2A: Ejecuta GeneLM sobre los fragmentos truncados (1, 2, 5, 10 kb)
en 3 modos:
  - e2_core_original: replica core.py (partial3=False, partial5=False, >100 coding nt)
  - e2_abiertos_gt100: candidatos completos + open_5 + open_3 + open_both (>100 nt en marco)
  - e2_abiertos_ge60: igual a e2_abiertos_gt100 pero con umbral >=60 nt en marco
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

_REPO_DIR = Path(__file__).resolve().parent.parent
_PARENT_DIR = _REPO_DIR.parent

import numpy as np
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_FRAG_FNA = (_REPO_DIR / "etapa2_datos_parciales/dataset_fragmentos/fragmentos.fna"
)
DEFAULT_MODELS = (_PARENT_DIR / "trabajos_referencia/datasets")
DEFAULT_OUT = (_REPO_DIR / "etapa2_datos_parciales/predicciones"
)

ALL_MODES = ("e2_core_original", "e2_abiertos_gt100", "e2_abiertos_ge60")
STARTS_CORE = ("TTG", "CTG", "ATG", "GTG")
STOPS_CORE = ("TAA", "TAG", "TGA")


def generate_kmer(sequence: str, k: int = 6, overlap: int = 1) -> str:
    if len(sequence) < k:
        return ""
    return " ".join(sequence[j : j + k] for j in range(0, len(sequence) - k + 1, overlap))


def extract_tis_window_original(
    seq_plus: str, seq_rc: str, strand: str, g_start0: int, g_end0: int, w: int = 30
) -> str:
    if strand == "+":
        return seq_plus[g_start0 - w : g_start0 + w]
    return seq_rc[-(g_end0 + w) : -(g_end0 - w)]


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


def oriented_to_genomic(strand: str, o_start: int, o_end: int, L: int) -> tuple[int, int]:
    if strand == "+":
        return o_start, o_end
    return L - o_end, L - o_start


def build_groups_for_fragment(frag_id: str, seq_u: str, mode: str) -> tuple[list[dict], dict]:
    """Construye los grupos de Etapa 1 para un fragmento según el modo."""
    seq_rc = str(Seq(seq_u).reverse_complement())
    L = len(seq_u)
    starts_set = set(STARTS_CORE)
    stops_set = set(STOPS_CORE)
    groups: list[dict] = []

    if mode == "e2_core_original":
        # Replicación exacta de core.py (partial3=False, partial5=False, coding_len > 100)
        grp_counter = {"+": 0, "-": 0}
        for strand, oriented in (("+", seq_u), ("-", seq_rc)):
            prefix = "P" if strand == "+" else "N"
            for frame in range(3):
                cur_starts: list[int] = []
                for pos in range(frame, L - 2, 3):
                    cod = oriented[pos : pos + 3]
                    if cod in stops_set:
                        end_local = pos + 3
                        cands = []
                        for st_pos in cur_starts:
                            if pos - st_pos >= 10:
                                g_s, g_e = oriented_to_genomic(strand, st_pos, end_local, L)
                                cands.append(
                                    {
                                        "g_start0": g_s,
                                        "g_end0": g_e,
                                        "o_start": st_pos,
                                        "o_end": end_local,
                                        "start_codon": oriented[st_pos : st_pos + 3],
                                        "stop_codon": cod,
                                        "coding_len": pos - st_pos,
                                        "total_len": end_local - st_pos,
                                    }
                                )
                        if cands:
                            longest = max(cands, key=lambda x: x["coding_len"])
                            if longest["coding_len"] > 100:
                                grp_counter[strand] += 1
                                orf_seq = oriented[longest["o_start"] : longest["o_end"]]
                                groups.append(
                                    {
                                        "fragment_id": frag_id,
                                        "group_uid": f"{frag_id}::{prefix}{grp_counter[strand]}",
                                        "strand": strand,
                                        "frame": frame,
                                        "region_kind": "complete",
                                        "span_g_start0": longest["g_start0"],
                                        "span_g_end0": longest["g_end0"],
                                        "span_o_start": longest["o_start"],
                                        "span_o_end": longest["o_end"],
                                        "in_frame_len": longest["total_len"],
                                        "stop_codon": cod,
                                        "kmer_cds": generate_kmer(orf_seq[-510:], k=6, overlap=3),
                                        "tis_candidates": cands,
                                    }
                                )
                        cur_starts = []
                    elif cod in starts_set:
                        cur_starts.append(pos)
        return groups, {"seq_u": seq_u, "seq_rc": seq_rc, "L": L}

    # Modos abiertos: e2_abiertos_gt100 y e2_abiertos_ge60
    min_inframe_len = 104 if mode == "e2_abiertos_gt100" else 60
    # Para complete/open_5 con codón stop, coding_len > 100 equivale a in_frame_len >= 105 (múltiplo de 3 > 103);
    # para open_3/open_both sin codón stop, > 100 nt equivale a >= 102 nt en múltiplos de 3.
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
                # 1) Primer stop del marco: cubre borde 5' (open_5) + inicios internos antes del primer stop
                first_stop_i = stop_indices[0]
                first_stop_end = codon_starts[first_stop_i] + 3
                in_frame_seq = oriented[frame:first_stop_end]
                in_frame_len = len(in_frame_seq)
                passes_len = (
                    (in_frame_len - 3 > 100)
                    if mode == "e2_abiertos_gt100"
                    else (in_frame_len >= min_inframe_len)
                )
                if passes_len:
                    grp_idx += 1
                    g_s_open5, g_e_open5 = oriented_to_genomic(strand, 0, first_stop_end, L)
                    cands = []
                    for si in start_indices:
                        if si >= first_stop_i:
                            break
                        st_pos = codon_starts[si]
                        if codon_starts[first_stop_i] - st_pos >= 10:
                            gs, ge = oriented_to_genomic(strand, st_pos, first_stop_end, L)
                            cands.append(
                                {
                                    "g_start0": gs,
                                    "g_end0": ge,
                                    "o_start": st_pos,
                                    "o_end": first_stop_end,
                                    "start_codon": codons[si],
                                    "stop_codon": codons[first_stop_i],
                                    "coding_len": codon_starts[first_stop_i] - st_pos,
                                    "total_len": first_stop_end - st_pos,
                                }
                            )
                    groups.append(
                        {
                            "fragment_id": frag_id,
                            "group_uid": f"{frag_id}::{prefix}{grp_idx}",
                            "strand": strand,
                            "frame": frame,
                            "region_kind": "open_5",
                            "span_g_start0": g_s_open5,
                            "span_g_end0": g_e_open5,
                            "span_o_start": 0,
                            "span_o_end": first_stop_end,
                            "in_frame_len": in_frame_len,
                            "stop_codon": codons[first_stop_i],
                            "kmer_cds": generate_kmer(in_frame_seq[-510:], k=6, overlap=3),
                            "tis_candidates": cands,
                        }
                    )

                # 2) Stops internos posteriores al primero (ORFs completos)
                for s_pos_list_idx in range(1, len(stop_indices)):
                    prev_si = stop_indices[s_pos_list_idx - 1]
                    cur_si = stop_indices[s_pos_list_idx]
                    stop_end = codon_starts[cur_si] + 3
                    cands = []
                    for si in start_indices:
                        if prev_si < si < cur_si:
                            st_pos = codon_starts[si]
                            if codon_starts[cur_si] - st_pos >= 10:
                                gs, ge = oriented_to_genomic(strand, st_pos, stop_end, L)
                                cands.append(
                                    {
                                        "g_start0": gs,
                                        "g_end0": ge,
                                        "o_start": st_pos,
                                        "o_end": stop_end,
                                        "start_codon": codons[si],
                                        "stop_codon": codons[cur_si],
                                        "coding_len": codon_starts[cur_si] - st_pos,
                                        "total_len": stop_end - st_pos,
                                    }
                                )
                    if not cands:
                        continue
                    longest = max(cands, key=lambda x: x["coding_len"])
                    passes_len = (
                        (longest["coding_len"] > 100)
                        if mode == "e2_abiertos_gt100"
                        else (longest["total_len"] >= min_inframe_len)
                    )
                    if not passes_len:
                        continue
                    grp_idx += 1
                    orf_seq = oriented[longest["o_start"] : longest["o_end"]]
                    groups.append(
                        {
                            "fragment_id": frag_id,
                            "group_uid": f"{frag_id}::{prefix}{grp_idx}",
                            "strand": strand,
                            "frame": frame,
                            "region_kind": "complete",
                            "span_g_start0": longest["g_start0"],
                            "span_g_end0": longest["g_end0"],
                            "span_o_start": longest["o_start"],
                            "span_o_end": longest["o_end"],
                            "in_frame_len": longest["total_len"],
                            "stop_codon": codons[cur_si],
                            "kmer_cds": generate_kmer(orf_seq[-510:], k=6, overlap=3),
                            "tis_candidates": cands,
                        }
                    )

                # 3) Región abierta en 3' tras el último stop (open_3)
                last_si = stop_indices[-1]
                after_starts = [si for si in start_indices if si > last_si]
                if after_starts:
                    cands = []
                    for si in after_starts:
                        st_pos = codon_starts[si]
                        if aligned_end - st_pos >= 12:
                            gs, ge = oriented_to_genomic(strand, st_pos, L, L)
                            cands.append(
                                {
                                    "g_start0": gs,
                                    "g_end0": ge,
                                    "o_start": st_pos,
                                    "o_end": L,
                                    "start_codon": codons[si],
                                    "stop_codon": "",
                                    "coding_len": aligned_end - st_pos,
                                    "total_len": L - st_pos,
                                }
                            )
                    if cands:
                        longest = max(cands, key=lambda x: x["coding_len"])
                        in_frame_seq = oriented[longest["o_start"] : aligned_end]
                        in_frame_len = len(in_frame_seq)
                        passes_len = (
                            (in_frame_len > 100)
                            if mode == "e2_abiertos_gt100"
                            else (in_frame_len >= min_inframe_len)
                        )
                        if passes_len:
                            grp_idx += 1
                            groups.append(
                                {
                                    "fragment_id": frag_id,
                                    "group_uid": f"{frag_id}::{prefix}{grp_idx}",
                                    "strand": strand,
                                    "frame": frame,
                                    "region_kind": "open_3",
                                    "span_g_start0": longest["g_start0"],
                                    "span_g_end0": longest["g_end0"],
                                    "span_o_start": longest["o_start"],
                                    "span_o_end": L,
                                    "in_frame_len": in_frame_len,
                                    "stop_codon": "",
                                    "kmer_cds": generate_kmer(in_frame_seq[-510:], k=6, overlap=3),
                                    "tis_candidates": cands,
                                }
                            )
            else:
                # 4) Sin ningún stop en todo el marco: open_both + posibles inicios internos (open_3)
                in_frame_seq = oriented[frame:aligned_end]
                in_frame_len = len(in_frame_seq)
                passes_len = (
                    (in_frame_len > 100)
                    if mode == "e2_abiertos_gt100"
                    else (in_frame_len >= min_inframe_len)
                )
                if passes_len:
                    grp_idx += 1
                    g_s_both, g_e_both = oriented_to_genomic(strand, 0, L, L)
                    cands = []
                    for si in start_indices:
                        st_pos = codon_starts[si]
                        if aligned_end - st_pos >= 12:
                            gs, ge = oriented_to_genomic(strand, st_pos, L, L)
                            cands.append(
                                {
                                    "g_start0": gs,
                                    "g_end0": ge,
                                    "o_start": st_pos,
                                    "o_end": L,
                                    "start_codon": codons[si],
                                    "stop_codon": "",
                                    "coding_len": aligned_end - st_pos,
                                    "total_len": L - st_pos,
                                }
                            )
                    groups.append(
                        {
                            "fragment_id": frag_id,
                            "group_uid": f"{frag_id}::{prefix}{grp_idx}",
                            "strand": strand,
                            "frame": frame,
                            "region_kind": "open_both",
                            "span_g_start0": g_s_both,
                            "span_g_end0": g_e_both,
                            "span_o_start": 0,
                            "span_o_end": L,
                            "in_frame_len": in_frame_len,
                            "stop_codon": "",
                            "kmer_cds": generate_kmer(in_frame_seq[-510:], k=6, overlap=3),
                            "tis_candidates": cands,
                        }
                    )

    return groups, {"seq_u": seq_u, "seq_rc": seq_rc, "L": L}


def predict_logits(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    sequences: list[str],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if not sequences:
        return np.zeros((0, 2), dtype=np.float32)
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
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
                    logits = model(**inputs).logits
            else:
                logits = model(**inputs).logits
            chunks.append(logits.float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def softmax_2d(logits: np.ndarray) -> np.ndarray:
    if len(logits) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)


def partial_attr(cand_type: str) -> str:
    return {
        "complete": "00",
        "open_5": "10",
        "open_3": "01",
        "open_both": "11",
    }[cand_type]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-fna", type=Path, default=DEFAULT_FRAG_FNA)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--modes", nargs="+", choices=ALL_MODES, default=list(ALL_MODES))
    parser.add_argument("--max-fragments", type=int, default=0, help="0 = todos")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    records = list(SeqIO.parse(str(args.input_fna), "fasta"))
    if args.max_fragments > 0:
        records = records[: args.max_fragments]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cds_dir = args.models_dir / "genelm_cds"
    tis_dir = args.models_dir / "genelm_tis"
    tokenizer = AutoTokenizer.from_pretrained(str(cds_dir), local_files_only=True)
    model_cds = AutoModelForSequenceClassification.from_pretrained(
        str(cds_dir), local_files_only=True
    ).to(device).eval()
    model_tis = AutoModelForSequenceClassification.from_pretrained(
        str(tis_dir), local_files_only=True
    ).to(device).eval()

    for mode in args.modes:
        mode_dir = args.out_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()

        frag_seqs: dict[str, dict] = {}
        all_groups: list[dict] = []
        region_counts = Counter()

        for rec in records:
            seq_u = str(rec.seq).upper()
            groups, seq_info = build_groups_for_fragment(rec.id, seq_u, mode)
            frag_seqs[rec.id] = seq_info
            all_groups.extend(groups)
            for g in groups:
                region_counts[g["region_kind"]] += 1

        # Etapa 1: BacteriaCDS por lotes sobre todos los grupos
        cds_kmers = [g["kmer_cds"] for g in all_groups]
        cds_logits = predict_logits(model_cds, tokenizer, cds_kmers, device, args.batch_size)
        cds_probs = softmax_2d(cds_logits)

        tis_inputs: list[dict] = []
        empty_tis = 0
        trunc_tis = 0
        cds_pos_counts = Counter()

        for g, lrow, prow in zip(all_groups, cds_logits, cds_probs):
            pred_cds = int(np.argmax(lrow))
            g["cds_prob1"] = float(prow[1])
            g["cds_pred"] = pred_cds
            g["emitted_type"] = ""
            if pred_cds != 1:
                continue
            cds_pos_counts[g["region_kind"]] += 1
            sinfo = frag_seqs[g["fragment_id"]]
            for cand in g["tis_candidates"]:
                if mode == "e2_core_original":
                    win = extract_tis_window_original(
                        sinfo["seq_u"], sinfo["seq_rc"], g["strand"], cand["g_start0"], cand["g_end0"]
                    )
                else:
                    win = extract_tis_window_safe(
                        sinfo["seq_u"], sinfo["seq_rc"], g["strand"], cand["g_start0"], cand["g_end0"]
                    )
                if len(win) == 0:
                    empty_tis += 1
                elif len(win) < 60:
                    trunc_tis += 1
                tis_inputs.append(
                    {
                        "group_uid": g["group_uid"],
                        "cand": cand,
                        "kmer_tis": generate_kmer(win, k=6, overlap=1),
                        "win_len": len(win),
                    }
                )

        # Etapa 2: BacteriaTIS por lotes sobre todos los sitios candidatos
        tis_kmers = [t["kmer_tis"] for t in tis_inputs]
        tis_logits = predict_logits(model_tis, tokenizer, tis_kmers, device, args.batch_size)
        tis_probs = softmax_2d(tis_logits)

        best_tis_by_group: dict[str, tuple[float, dict, int]] = {}
        for item, prow in zip(tis_inputs, tis_probs):
            p0, p1 = float(prow[0]), float(prow[1])
            if p0 <= p1:  # pred_classic == 1
                guid = item["group_uid"]
                if guid not in best_tis_by_group or p1 > best_tis_by_group[guid][0]:
                    best_tis_by_group[guid] = (p1, item["cand"], item["win_len"])

        # Resolución final y construcción de predicciones GFF3
        final_calls: list[dict] = []
        emitted_counts = Counter()

        for g in all_groups:
            if g["cds_pred"] != 1:
                continue
            guid = g["group_uid"]
            rk = g["region_kind"]
            best = best_tis_by_group.get(guid)

            if rk == "complete":
                if best is None:
                    continue
                tis_p1, cand, win_len = best
                cand_type = "complete"
                gs, ge = cand["g_start0"], cand["g_end0"]
                sc, tc = cand["start_codon"], cand["stop_codon"]
            elif rk == "open_5":
                if best is not None:
                    tis_p1, cand, win_len = best
                    cand_type = "complete"
                    gs, ge = cand["g_start0"], cand["g_end0"]
                    sc, tc = cand["start_codon"], cand["stop_codon"]
                else:
                    cand_type = "open_5"
                    tis_p1, win_len = g["cds_prob1"], 0
                    gs, ge = g["span_g_start0"], g["span_g_end0"]
                    sc, tc = "", g["stop_codon"]
            elif rk == "open_3":
                if best is None:
                    continue
                tis_p1, cand, win_len = best
                cand_type = "open_3"
                gs, ge = cand["g_start0"], cand["g_end0"]
                sc, tc = cand["start_codon"], ""
            elif rk == "open_both":
                if best is not None:
                    tis_p1, cand, win_len = best
                    cand_type = "open_3"
                    gs, ge = cand["g_start0"], cand["g_end0"]
                    sc, tc = cand["start_codon"], ""
                else:
                    cand_type = "open_both"
                    tis_p1, win_len = g["cds_prob1"], 0
                    gs, ge = g["span_g_start0"], g["span_g_end0"]
                    sc, tc = "", ""
            else:
                continue

            frag_L = frag_seqs[g["fragment_id"]]["L"]
            assert 0 <= gs < ge <= frag_L, f"Invalid coords {gs}..{ge} for {g['fragment_id']}"
            g["emitted_type"] = cand_type
            emitted_counts[cand_type] += 1
            final_calls.append(
                {
                    "fragment_id": g["fragment_id"],
                    "strand": g["strand"],
                    "start1": gs + 1,
                    "end1": ge,
                    "score": tis_p1,
                    "cds_prob1": g["cds_prob1"],
                    "cand_type": cand_type,
                    "partial": partial_attr(cand_type),
                    "start_codon": sc,
                    "stop_codon": tc,
                    "win_len": win_len,
                    "group_uid": guid,
                }
            )

        final_calls.sort(key=lambda r: (r["fragment_id"], r["start1"], r["end1"], r["strand"]))

        # Escribir predicciones.gff
        gff_path = mode_dir / "predicciones.gff"
        with gff_path.open("w", encoding="utf-8") as fh:
            fh.write("##gff-version 3\n")
            fh.write(f"##Tool: GeneLM-local ({mode})\n")
            for idx, c in enumerate(final_calls, start=1):
                attrs = (
                    f"ID={mode}_{idx:06d};cand_type={c['cand_type']};"
                    f"partial={c['partial']};cds_prob1={c['cds_prob1']:.4f};"
                    f"score={c['score']:.4f};start_codon={c['start_codon']};"
                    f"stop_codon={c['stop_codon']};tis_window_len={c['win_len']}"
                )
                fh.write(
                    "\t".join(
                        [
                            c["fragment_id"],
                            f"GeneLM_{mode}",
                            "CDS",
                            str(c["start1"]),
                            str(c["end1"]),
                            f"{c['score']:.4f}",
                            c["strand"],
                            "0",
                            attrs,
                        ]
                    )
                    + "\n"
                )

        # Escribir etapa1_grupos.tsv
        grp_path = mode_dir / "etapa1_grupos.tsv"
        grp_fields = [
            "fragment_id",
            "group_uid",
            "strand",
            "frame",
            "region_kind",
            "span_start0",
            "span_end0",
            "in_frame_len",
            "cds_prob1",
            "cds_pred",
            "emitted_type",
        ]
        with grp_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=grp_fields, delimiter="\t")
            w.writeheader()
            for g in all_groups:
                w.writerow(
                    {
                        "fragment_id": g["fragment_id"],
                        "group_uid": g["group_uid"],
                        "strand": g["strand"],
                        "frame": g["frame"],
                        "region_kind": g["region_kind"],
                        "span_start0": g["span_g_start0"],
                        "span_end0": g["span_g_end0"],
                        "in_frame_len": g["in_frame_len"],
                        "cds_prob1": round(g["cds_prob1"], 6),
                        "cds_pred": g["cds_pred"],
                        "emitted_type": g["emitted_type"],
                    }
                )

        elapsed = time.perf_counter() - t0
        peak_mib = (
            round(torch.cuda.max_memory_allocated(device) / (1024**2), 2)
            if device.type == "cuda"
            else 0.0
        )
        stats = {
            "mode": mode,
            "fragments": len(records),
            "total_bp": sum(frag_seqs[r.id]["L"] for r in records),
            "stage1_groups_by_region_kind": dict(region_counts),
            "stage1_pos_by_region_kind": dict(cds_pos_counts),
            "tis_candidates_evaluated": len(tis_inputs),
            "empty_tis_windows": empty_tis,
            "truncated_tis_windows": trunc_tis,
            "emitted_calls_total": len(final_calls),
            "emitted_calls_by_cand_type": dict(emitted_counts),
            "elapsed_seconds": round(elapsed, 3),
            "peak_vram_mib": peak_mib,
        }
        (mode_dir / "runtime_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        print(
            f"[{mode}] frags={len(records)} groups={len(all_groups)} "
            f"cds_pos={sum(cds_pos_counts.values())} calls={len(final_calls)} "
            f"by_type={dict(emitted_counts)} empty_tis={empty_tis} "
            f"time={stats['elapsed_seconds']}s vram={peak_mib} MiB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
