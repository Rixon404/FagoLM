#!/usr/bin/env python3
"""Ejecuta inferencia de Etapa 1 (fagos completos) con los pesos locales de GeneLM
en 3 modos diagnósticos controlados sobre el piloto INPHARED.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import defaultdict
from pathlib import Path

_REPO_DIR = Path(__file__).resolve().parent.parent
_PARENT_DIR = _REPO_DIR.parent

import numpy as np
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_PILOTO = (_REPO_DIR / "etapa1_fagos_completos/piloto_inphared"
)
DEFAULT_OUT = (_REPO_DIR / "etapa1_fagos_completos/predicciones"
)
DEFAULT_MODELS = (_PARENT_DIR / "trabajos_referencia/datasets")

ALL_MODES = (
    "modo_a_original",
    "modo_b_adaptado_fagos",
    "modo_c_diagnostico_hebra_menos",
)


def generate_kmer(sequence: str, k: int = 6, overlap: int = 1) -> str:
    if len(sequence) < k:
        return ""
    return " ".join(sequence[j : j + k] for j in range(0, len(sequence) - k + 1, overlap))


def extract_complete_nested_orfs(
    dna: str,
    starts: tuple[str, ...] | list[str],
    stops: tuple[str, ...] | list[str],
    min_coding_len: int = 10,
) -> tuple[list[dict], list[dict]]:
    """Replica exactamente oc.start_search(..., minlen=..., partial3=False, partial5=False, nested=True, include_stop=True).

    Retorna (orfs_pos, orfs_neg) con coordenadas genómicas 0-based half-open [start, end),
    'length' igual a coding_len (excluyendo stop, igual que orfipy), 'total_len' = end - start,
    y 'sequence' en orientación 5'->3' incluyendo el codón de parada.
    """
    dna_u = dna.upper()
    dna_rc = str(Seq(dna_u).reverse_complement())
    seq_len = len(dna_u)
    starts_set = set(starts)
    stops_set = set(stops)

    orfs_pos: list[dict] = []
    orfs_neg: list[dict] = []

    for strand, seq in (("+", dna_u), ("-", dna_rc)):
        target = orfs_pos if strand == "+" else orfs_neg
        for offset in range(3):
            cur_starts: list[int] = []
            for pos in range(offset, seq_len - 2, 3):
                codon = seq[pos : pos + 3]
                if codon in stops_set:
                    end_local = pos + 3
                    for st_pos in cur_starts:
                        coding_len = pos - st_pos
                        if coding_len >= min_coding_len:
                            if strand == "+":
                                g_start, g_end = st_pos, end_local
                            else:
                                g_start, g_end = seq_len - end_local, seq_len - st_pos
                            target.append(
                                {
                                    "start": g_start,
                                    "end": g_end,
                                    "strand": strand,
                                    "length": coding_len,
                                    "total_len": end_local - st_pos,
                                    "start_codon": seq[st_pos : st_pos + 3],
                                    "stop_codon": codon,
                                    "sequence": seq[st_pos:end_local],
                                }
                            )
                    cur_starts = []
                elif codon in starts_set:
                    cur_starts.append(pos)
    return orfs_pos, orfs_neg


def verify_orfipy_parity(fna_path: Path) -> dict:
    """Compara el extractor en Python puro contra orfipy_core.start_search en el primer genoma."""
    try:
        import orfipy_core as oc
    except ImportError as exc:
        return {"available": False, "error": str(exc)}

    rec = next(SeqIO.parse(str(fna_path), "fasta"))
    seq = str(rec.seq).upper()
    seq_rc = str(Seq(seq).reverse_complement())

    res = oc.start_search(
        seq,
        seq_rc,
        rec.id,
        10,
        10_000_000,
        "b",
        ["TTG", "CTG", "ATG", "GTG"],
        ["TAA", "TAG", "TGA"],
        "1",
        True,
        False,
        False,
        False,
        True,
        [False, False, True, False, False],
    )
    oc_set = set()
    for entry in res[2].strip().split(">")[1:]:
        lines = entry.splitlines()
        m = re.search(
            r"(\d+)-(\d+)\]\((\+|-)\) type:complete length:(\d+) frame:(-?\d+) start:(\w{3}) stop:(\w{3})",
            lines[0],
        )
        if not m:
            continue
        oc_set.add(
            (
                int(m.group(1)),
                int(m.group(2)),
                m.group(3),
                int(m.group(4)),
                m.group(6),
                m.group(7),
                len("".join(lines[1:])),
            )
        )

    pos, neg = extract_complete_nested_orfs(
        seq, ["TTG", "CTG", "ATG", "GTG"], ["TAA", "TAG", "TGA"], min_coding_len=10
    )
    py_set = {
        (o["start"], o["end"], o["strand"], o["length"], o["start_codon"], o["stop_codon"], o["total_len"])
        for o in (pos + neg)
    }
    return {
        "available": True,
        "genome": rec.id,
        "orfipy_count": len(oc_set),
        "python_count": len(py_set),
        "exact_match": oc_set == py_set,
    }


def predict_logits(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    sequences: list[str],
    device: torch.device,
    batch_size: int,
    use_fp16: bool,
) -> np.ndarray:
    if not sequences:
        return np.zeros((0, 2), dtype=np.float32)

    all_logits: list[np.ndarray] = []
    with torch.inference_mode():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i : i + batch_size]
            inputs = tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=tokenizer.model_max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            if device.type == "cuda" and use_fp16:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            all_logits.append(outputs.logits.float().cpu().numpy())
    return np.concatenate(all_logits, axis=0)


def softmax_2d(logits: np.ndarray) -> np.ndarray:
    if len(logits) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)


def extract_tis_window_original(seq_plus: str, seq_rc: str, strand: str, start_0: int, end_0: int, w: int = 30) -> str:
    """Replica exactamente core.py:221,239 (incluyendo fallos de borde < 30 pb)."""
    if strand == "+":
        return seq_plus[start_0 - w : start_0 + w]
    return seq_rc[-(end_0 + w) : -(end_0 - w)]


def extract_tis_window_safe(seq_plus: str, seq_rc: str, strand: str, start_0: int, end_0: int, w: int = 30) -> str:
    """Ventana de 60 pb centrada en el codón de inicio con padding 'N' si sale del borde."""
    L = len(seq_plus)
    if strand == "+":
        center = start_0
        src = seq_plus
    else:
        center = L - end_0
        src = seq_rc

    left = center - w
    right = center + w
    pad_left = "N" * max(0, -left)
    pad_right = "N" * max(0, right - L)
    core_sub = src[max(0, left) : min(L, right)]
    return pad_left + core_sub + pad_right


def prepare_cds_groups(
    orfs: list[dict],
    strand: str,
    mode: str,
    max_seq_len: int = 510,
) -> list[dict]:
    """Agrupa ORFs completos por codón de parada (misma lógica de core.py:_cds_input_parser)."""
    if not orfs:
        return []
    key_name = "end" if strand == "+" else "start"
    grouped: dict[int, list[dict]] = defaultdict(list)
    for o in orfs:
        grouped[o[key_name]].append(o)

    groups_out: list[dict] = []
    prefix = "P" if strand == "+" else "N"
    idx = 0
    for stop_coord in sorted(grouped):
        cand_list = grouped[stop_coord]
        longest = max(cand_list, key=lambda x: x["length"])
        if mode == "modo_b_adaptado_fagos":
            if longest["total_len"] < 60:
                continue
        else:
            if longest["length"] <= 100:
                continue

        idx += 1
        dna_seq = longest["sequence"]
        if mode == "modo_c_diagnostico_hebra_menos" and strand == "-":
            sliced = dna_seq[:max_seq_len]
        else:
            sliced = dna_seq[-max_seq_len:]

        # En modo_b solo conservamos inicios candidatos >= 60 nt totales
        if mode == "modo_b_adaptado_fagos":
            valid_cands = [c for c in cand_list if c["total_len"] >= 60]
        else:
            valid_cands = cand_list

        groups_out.append(
            {
                "orf_group": f"{prefix}{idx}",
                "strand": strand,
                "stop_coord_1": longest["end"] if strand == "+" else longest["start"] + 1,
                "longest_start": longest["start"],
                "longest_end": longest["end"],
                "longest_total_len": longest["total_len"],
                "longest_coding_len": longest["length"],
                "start_codon": longest["start_codon"],
                "stop_codon": longest["stop_codon"],
                "kmer_seq": generate_kmer(sliced, k=6, overlap=3),
                "candidates": sorted(
                    valid_cands,
                    key=lambda x: x["start"] if strand == "+" else -x["end"],
                ),
            }
        )
    return groups_out


def run_genome_mode(
    record_id: str,
    seq: str,
    mode: str,
    model_cds: AutoModelForSequenceClassification,
    model_tis: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: torch.device,
    batch_size: int,
    use_fp16: bool,
) -> dict:
    seq_u = seq.upper()
    seq_rc = str(Seq(seq_u).reverse_complement())

    if mode == "modo_b_adaptado_fagos":
        starts = ["ATG", "GTG", "TTG", "ATT", "ATC", "ATA"]
        min_coding_len = 57  # total_len >= 60 nt
    else:
        starts = ["TTG", "CTG", "ATG", "GTG"]
        min_coding_len = 10  # igual que core.py:340

    orfs_pos, orfs_neg = extract_complete_nested_orfs(
        seq_u, starts=starts, stops=["TAA", "TAG", "TGA"], min_coding_len=min_coding_len
    )

    groups = prepare_cds_groups(orfs_pos, "+", mode) + prepare_cds_groups(orfs_neg, "-", mode)
    cds_kmers = [g["kmer_seq"] for g in groups]
    cds_logits = predict_logits(model_cds, tokenizer, cds_kmers, device, batch_size, use_fp16)
    cds_probs = softmax_2d(cds_logits)

    tis_candidates: list[dict] = []
    empty_tis_windows = 0
    truncated_tis_windows = 0

    for g, logits_row, probs_row in zip(groups, cds_logits, cds_probs):
        pred_cds = int(np.argmax(logits_row))
        g["cds_logit0"] = float(logits_row[0])
        g["cds_logit1"] = float(logits_row[1])
        g["cds_prob0"] = float(probs_row[0])
        g["cds_prob1"] = float(probs_row[1])
        g["cds_pred"] = pred_cds
        g["selected_tis_start_1"] = ""
        g["selected_tis_end_1"] = ""
        g["selected_tis_prob1"] = ""

        if pred_cds != 1:
            continue

        for cand in g["candidates"]:
            if mode == "modo_b_adaptado_fagos":
                win = extract_tis_window_safe(
                    seq_u, seq_rc, cand["strand"], cand["start"], cand["end"], w=30
                )
            else:
                win = extract_tis_window_original(
                    seq_u, seq_rc, cand["strand"], cand["start"], cand["end"], w=30
                )
            if len(win) == 0:
                empty_tis_windows += 1
            elif len(win) < 60:
                truncated_tis_windows += 1

            tis_candidates.append(
                {
                    "record_id": record_id,
                    "orf_group": g["orf_group"],
                    "strand": cand["strand"],
                    "start_0": cand["start"],
                    "end_0": cand["end"],
                    "start_1": cand["start"] + 1,
                    "end_1": cand["end"],
                    "total_len": cand["total_len"],
                    "start_codon": cand["start_codon"],
                    "stop_codon": cand["stop_codon"],
                    "tis_window_len": len(win),
                    "kmer_seq": generate_kmer(win, k=6, overlap=1),
                    "cds_prob1": g["cds_prob1"],
                }
            )

    tis_kmers = [c["kmer_seq"] for c in tis_candidates]
    tis_logits = predict_logits(model_tis, tokenizer, tis_kmers, device, batch_size, use_fp16)
    tis_probs = softmax_2d(tis_logits)

    best_by_group: dict[str, tuple[float, int]] = {}
    for idx, (cand, logits_row, probs_row) in enumerate(zip(tis_candidates, tis_logits, tis_probs)):
        prob0, prob1 = float(probs_row[0]), float(probs_row[1])
        pred_classic = 0 if prob0 > prob1 else 1
        cand["tis_logit0"] = float(logits_row[0])
        cand["tis_logit1"] = float(logits_row[1])
        cand["tis_prob0"] = prob0
        cand["tis_prob1"] = prob1
        cand["tis_pred_classic"] = pred_classic
        cand["is_selected"] = 0
        if pred_classic == 1:
            grp = cand["orf_group"]
            if grp not in best_by_group or prob1 > best_by_group[grp][0]:
                best_by_group[grp] = (prob1, idx)

    selected_calls: list[dict] = []
    group_map = {g["orf_group"]: g for g in groups}
    for grp, (_, best_idx) in best_by_group.items():
        cand = tis_candidates[best_idx]
        cand["is_selected"] = 1
        selected_calls.append(cand)
        g = group_map[grp]
        g["selected_tis_start_1"] = cand["start_1"]
        g["selected_tis_end_1"] = cand["end_1"]
        g["selected_tis_prob1"] = round(cand["tis_prob1"], 6)

    selected_calls.sort(key=lambda r: (r["start_1"], r["end_1"], r["strand"]))

    return {
        "record_id": record_id,
        "orfs_pos": len(orfs_pos),
        "orfs_neg": len(orfs_neg),
        "cds_groups": groups,
        "tis_candidates": tis_candidates,
        "selected_calls": selected_calls,
        "empty_tis_windows": empty_tis_windows,
        "truncated_tis_windows": truncated_tis_windows,
    }


def write_genome_outputs(acc: str, out_mode_dir: Path, res: dict, mode: str) -> None:
    gff_path = out_mode_dir / f"{acc}.gff"
    with gff_path.open("w", encoding="utf-8") as fh:
        fh.write("##gff-version 3\n")
        fh.write(f"##Tool: GeneLM-local ({mode})\n")
        for idx, c in enumerate(res["selected_calls"], start=1):
            attrs = (
                f"ID={acc}_{idx:05d};orf_group={c['orf_group']};"
                f"cds_prob1={c['cds_prob1']:.4f};tis_prob1={c['tis_prob1']:.4f};"
                f"start_codon={c['start_codon']};stop_codon={c['stop_codon']};"
                f"tis_window_len={c['tis_window_len']}"
            )
            fh.write(
                "\t".join(
                    [
                        c["record_id"],
                        f"GeneLM_{mode}",
                        "CDS",
                        str(c["start_1"]),
                        str(c["end_1"]),
                        f"{c['tis_prob1']:.4f}",
                        c["strand"],
                        "0",
                        attrs,
                    ]
                )
                + "\n"
            )

    grp_path = out_mode_dir / f"{acc}_etapa1_cds_groups.tsv"
    grp_fields = [
        "orf_group",
        "strand",
        "stop_coord_1",
        "longest_start_1",
        "longest_end_1",
        "longest_total_len",
        "longest_coding_len",
        "start_codon",
        "stop_codon",
        "n_candidate_starts",
        "cds_prob0",
        "cds_prob1",
        "cds_pred",
        "selected_tis_start_1",
        "selected_tis_end_1",
        "selected_tis_prob1",
    ]
    with grp_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=grp_fields, delimiter="\t")
        w.writeheader()
        for g in res["cds_groups"]:
            w.writerow(
                {
                    "orf_group": g["orf_group"],
                    "strand": g["strand"],
                    "stop_coord_1": g["stop_coord_1"],
                    "longest_start_1": g["longest_start"] + 1,
                    "longest_end_1": g["longest_end"],
                    "longest_total_len": g["longest_total_len"],
                    "longest_coding_len": g["longest_coding_len"],
                    "start_codon": g["start_codon"],
                    "stop_codon": g["stop_codon"],
                    "n_candidate_starts": len(g["candidates"]),
                    "cds_prob0": round(g["cds_prob0"], 6),
                    "cds_prob1": round(g["cds_prob1"], 6),
                    "cds_pred": g["cds_pred"],
                    "selected_tis_start_1": g["selected_tis_start_1"],
                    "selected_tis_end_1": g["selected_tis_end_1"],
                    "selected_tis_prob1": g["selected_tis_prob1"],
                }
            )

    tis_path = out_mode_dir / f"{acc}_etapa2_tis_candidates.tsv"
    tis_fields = [
        "record_id",
        "orf_group",
        "strand",
        "start_1",
        "end_1",
        "total_len",
        "start_codon",
        "stop_codon",
        "tis_window_len",
        "cds_prob1",
        "tis_prob0",
        "tis_prob1",
        "tis_pred_classic",
        "is_selected",
    ]
    with tis_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=tis_fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(res["tis_candidates"])


def check_fp16_parity(
    first_fna: Path,
    model_cds: AutoModelForSequenceClassification,
    model_tis: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: torch.device,
    batch_size: int,
) -> dict:
    rec = next(SeqIO.parse(str(first_fna), "fasta"))
    res_fp32 = run_genome_mode(
        rec.id, str(rec.seq), "modo_a_original", model_cds, model_tis, tokenizer, device, min(batch_size, 16), use_fp16=False
    )
    res_fp16 = run_genome_mode(
        rec.id, str(rec.seq), "modo_a_original", model_cds, model_tis, tokenizer, device, batch_size, use_fp16=True
    )
    cds32 = [g["cds_pred"] for g in res_fp32["cds_groups"]]
    cds16 = [g["cds_pred"] for g in res_fp16["cds_groups"]]
    cds_agree = sum(a == b for a, b in zip(cds32, cds16)) / len(cds32) if cds32 else 1.0

    map32 = {(c["start_1"], c["end_1"], c["strand"]): c["tis_pred_classic"] for c in res_fp32["tis_candidates"]}
    map16 = {(c["start_1"], c["end_1"], c["strand"]): c["tis_pred_classic"] for c in res_fp16["tis_candidates"]}
    common = set(map32) & set(map16)
    tis_agree = sum(map32[k] == map16[k] for k in common) / len(common) if common else 1.0

    sel32 = {(c["start_1"], c["end_1"], c["strand"]) for c in res_fp32["selected_calls"]}
    sel16 = {(c["start_1"], c["end_1"], c["strand"]) for c in res_fp16["selected_calls"]}
    sel_exact = sel32 == sel16
    return {
        "genome": rec.id,
        "cds_groups_count": len(cds32),
        "cds_class_agreement": round(cds_agree, 6),
        "tis_common_candidates": len(common),
        "tis_class_agreement": round(tis_agree, 6),
        "selected_calls_fp32": len(sel32),
        "selected_calls_fp16": len(sel16),
        "selected_calls_identical": sel_exact,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--piloto-dir", type=Path, default=DEFAULT_PILOTO)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--modes", nargs="+", choices=ALL_MODES, default=list(ALL_MODES))
    parser.add_argument("--max-genomes", type=int, default=0, help="0 = todos los del manifiesto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dtype", choices=["auto", "fp16", "fp32"], default="auto")
    args = parser.parse_args()

    manifest_path = args.piloto_dir / "manifiesto_piloto.tsv"
    with manifest_path.open(encoding="utf-8") as fh:
        manifest_rows = list(csv.DictReader(fh, delimiter="\t"))
    if args.max_genomes > 0:
        manifest_rows = manifest_rows[: args.max_genomes]

    first_fna = args.piloto_dir / "genomas" / f"{manifest_rows[0]['accession']}.fna"
    parity_orf = verify_orfipy_parity(first_fna)
    print("Verificación paridad extractor ORFs vs orfipy_core:", json.dumps(parity_orf))
    if parity_orf.get("available") and not parity_orf.get("exact_match"):
        raise RuntimeError("Fallo de paridad entre extractor puro y orfipy_core")

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

    fp16_check = None
    if device.type == "cuda":
        if args.dtype == "fp16":
            use_fp16 = True
        elif args.dtype == "fp32":
            use_fp16 = False
        else:
            fp16_check = check_fp16_parity(
                first_fna, model_cds, model_tis, tokenizer, device, args.batch_size
            )
            print("Verificación paridad FP16 vs FP32:", json.dumps(fp16_check))
            use_fp16 = (
                fp16_check["cds_class_agreement"] >= 0.995
                and fp16_check["tis_class_agreement"] >= 0.995
            )
    else:
        use_fp16 = False

    effective_batch = args.batch_size if use_fp16 else min(args.batch_size, 16)
    print(f"Dispositivo: {device} | FP16: {use_fp16} | batch_size: {effective_batch}")

    for mode in args.modes:
        out_mode_dir = args.out_dir / mode
        out_mode_dir.mkdir(parents=True, exist_ok=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()

        totals = {
            "mode": mode,
            "genomes": len(manifest_rows),
            "device": str(device),
            "use_fp16": use_fp16,
            "batch_size": effective_batch,
            "orfipy_parity": parity_orf,
            "fp16_parity_check": fp16_check,
            "total_orfs_pos": 0,
            "total_orfs_neg": 0,
            "total_cds_groups": 0,
            "total_cds_pos_pred": 0,
            "total_tis_candidates": 0,
            "total_empty_tis_windows": 0,
            "total_truncated_tis_windows": 0,
            "total_selected_calls": 0,
            "per_genome": [],
        }

        for row in manifest_rows:
            acc = row["accession"]
            fna_path = args.piloto_dir / "genomas" / f"{acc}.fna"
            rec = next(SeqIO.parse(str(fna_path), "fasta"))
            g_t0 = time.perf_counter()
            res = run_genome_mode(
                rec.id,
                str(rec.seq),
                mode,
                model_cds,
                model_tis,
                tokenizer,
                device,
                effective_batch,
                use_fp16,
            )
            g_dt = time.perf_counter() - g_t0
            write_genome_outputs(acc, out_mode_dir, res, mode)

            n_cds_pos = sum(1 for g in res["cds_groups"] if g["cds_pred"] == 1)
            totals["total_orfs_pos"] += res["orfs_pos"]
            totals["total_orfs_neg"] += res["orfs_neg"]
            totals["total_cds_groups"] += len(res["cds_groups"])
            totals["total_cds_pos_pred"] += n_cds_pos
            totals["total_tis_candidates"] += len(res["tis_candidates"])
            totals["total_empty_tis_windows"] += res["empty_tis_windows"]
            totals["total_truncated_tis_windows"] += res["truncated_tis_windows"]
            totals["total_selected_calls"] += len(res["selected_calls"])
            totals["per_genome"].append(
                {
                    "accession": acc,
                    "length_bp": len(rec.seq),
                    "cds_groups": len(res["cds_groups"]),
                    "cds_pos_pred": n_cds_pos,
                    "tis_candidates": len(res["tis_candidates"]),
                    "empty_tis_windows": res["empty_tis_windows"],
                    "truncated_tis_windows": res["truncated_tis_windows"],
                    "selected_calls": len(res["selected_calls"]),
                    "seconds": round(g_dt, 3),
                }
            )

        elapsed = time.perf_counter() - t0
        peak_mib = (
            round(torch.cuda.max_memory_allocated(device) / (1024**2), 2)
            if device.type == "cuda"
            else 0.0
        )
        totals["elapsed_seconds"] = round(elapsed, 3)
        totals["seconds_per_genome"] = round(elapsed / len(manifest_rows), 3)
        totals["peak_vram_mib"] = peak_mib

        stats_path = out_mode_dir / "runtime_stats.json"
        stats_path.write_text(json.dumps(totals, indent=2) + "\n", encoding="utf-8")
        print(
            f"[{mode}] genomas={len(manifest_rows)} | cds_groups={totals['total_cds_groups']} | "
            f"cds_pred1={totals['total_cds_pos_pred']} | calls={totals['total_selected_calls']} | "
            f"empty_tis={totals['total_empty_tis_windows']} | trunc_tis={totals['total_truncated_tis_windows']} | "
            f"time={totals['elapsed_seconds']}s | peak_vram={peak_mib} MiB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
