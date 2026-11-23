#!/usr/bin/env python3
"""Benchmark 2B: Prueba controlada directa sobre BacteriaCDS-DNABERT-K6-89M ante
cortes exactos en marco (sin 5', sin 3', sin ambos y truncado corto <=100 nt).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

_REPO_DIR = Path(__file__).resolve().parent.parent
_PARENT_DIR = _REPO_DIR.parent

import numpy as np
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_PILOTO = (_REPO_DIR / "etapa1_fagos_completos/piloto_inphared"
)
DEFAULT_MODELS = (_PARENT_DIR / "trabajos_referencia/datasets")
DEFAULT_OUT = (_REPO_DIR / "etapa2_datos_parciales/benchmark_directo_cds"
)


def generate_kmer(sequence: str, k: int = 6, overlap: int = 3) -> str:
    if len(sequence) < k:
        return ""
    return " ".join(sequence[j : j + k] for j in range(0, len(sequence) - k + 1, overlap))


def extract_complete_orfs_longest_per_stop(dna_u: str) -> list[dict]:
    """Replica agrupación de modo_a_original: starts TTG/CTG/ATG/GTG, stops TAA/TAG/TGA, coding_len > 100."""
    starts = {"TTG", "CTG", "ATG", "GTG"}
    stops = {"TAA", "TAG", "TGA"}
    dna_rc = str(Seq(dna_u).reverse_complement())
    seq_len = len(dna_u)
    best_by_stop: dict[tuple[str, int], dict] = {}

    for strand, seq in (("+", dna_u), ("-", dna_rc)):
        for offset in range(3):
            cur_starts: list[int] = []
            for pos in range(offset, seq_len - 2, 3):
                cod = seq[pos : pos + 3]
                if cod in stops:
                    end_local = pos + 3
                    if cur_starts:
                        first_st = cur_starts[0]
                        coding_len = pos - first_st
                        if coding_len > 100:
                            if strand == "+":
                                stop_coord = end_local
                            else:
                                stop_coord = seq_len - end_local
                            best_by_stop[(strand, stop_coord)] = {
                                "strand": strand,
                                "stop_coord_0": stop_coord,
                                "sequence": seq[first_st:end_local],
                            }
                    cur_starts = []
                elif cod in starts:
                    cur_starts.append(pos)
    return list(best_by_stop.values())


def parse_pos_cds(gff_path: Path, seq_u: str, record_id: str) -> list[dict]:
    pos_rows: list[dict] = []
    for raw in gff_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            if line == "##FASTA":
                break
            continue
        parts = line.split("\t")
        if len(parts) != 9 or parts[2].lower() != "cds":
            continue
        seqid, _src, _ft, s1_s, e1_s, _sc, strand, _ph, _attrs = parts
        if seqid != record_id or strand not in {"+", "-"}:
            continue
        s0, e0 = int(s1_s) - 1, int(e1_s)
        sub = seq_u[s0:e0]
        if strand == "-":
            sub = str(Seq(sub).reverse_complement())
            stop_coord = s0
        else:
            stop_coord = e0
        pos_rows.append(
            {
                "record_id": record_id,
                "strand": strand,
                "stop_coord_0": stop_coord,
                "sequence": sub,
                "label": 1,
            }
        )
    return pos_rows


def length_stratum(n: int) -> str:
    if n <= 300:
        return "<=300"
    if n <= 600:
        return "301-600"
    if n <= 900:
        return "601-900"
    return ">900"


def sample_matched_negatives(pos_rows: list[dict], neg_pool: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    pos_bins = Counter(length_stratum(len(r["sequence"])) for r in pos_rows)
    neg_by_bin: dict[str, list[dict]] = defaultdict(list)
    for r in neg_pool:
        neg_by_bin[length_stratum(len(r["sequence"]))].append(r)
    for b in neg_by_bin:
        neg_by_bin[b].sort(key=lambda x: (x["record_id"], x["strand"], x["stop_coord_0"]))
        rng.shuffle(neg_by_bin[b])

    chosen: list[dict] = []
    chosen_keys: set[tuple[str, str, int]] = set()
    for b in ("<=300", "301-600", "601-900", ">900"):
        take = min(pos_bins[b], len(neg_by_bin[b]))
        for item in neg_by_bin[b][:take]:
            chosen.append(item)
            chosen_keys.add((item["record_id"], item["strand"], item["stop_coord_0"]))

    remaining_needed = len(pos_rows) - len(chosen)
    if remaining_needed > 0:
        leftover = [
            item
            for item in sorted(neg_pool, key=lambda x: (x["record_id"], x["strand"], x["stop_coord_0"]))
            if (item["record_id"], item["strand"], item["stop_coord_0"]) not in chosen_keys
        ]
        rng.shuffle(leftover)
        chosen.extend(leftover[:remaining_needed])
    return chosen


def transform_sequence(seq: str, condition: str) -> str | None:
    L = len(seq)
    if L % 3 != 0:
        return None
    n_cod = L // 3

    if condition == "complete_gt103":
        if L < 105:
            return None
        return seq[-510:]

    if condition == "sin_5prime_50pct":
        half_cod = n_cod // 2
        start_cod = n_cod - half_cod
        if start_cod < 1:
            return None
        sub = seq[start_cod * 3 :]
        if len(sub) < 105:
            return None
        return sub[-510:]

    if condition in ("sin_3prime_50pct_slice3p", "sin_3prime_50pct_slice5p"):
        half_cod = n_cod // 2
        if half_cod >= n_cod:
            return None
        sub = seq[: half_cod * 3]
        if len(sub) < 105:
            return None
        return sub[-510:] if condition == "sin_3prime_50pct_slice3p" else sub[:510]

    if condition == "sin_ambos_50pct":
        q = max(1, n_cod // 4)
        c_start = q * 3
        c_end = (n_cod - q) * 3
        if c_start < 3 or c_end > L - 3 or c_end <= c_start:
            return None
        sub = seq[c_start:c_end]
        if len(sub) < 105:
            return None
        return sub[-510:]

    if condition == "truncado_corto_le100":
        # 75 nt internos en marco (25 codones), sin codón de inicio ni stop
        if L < 81:
            return None
        mid_cod = n_cod // 2
        c_start_cod = max(1, mid_cod - 12)
        c_end_cod = c_start_cod + 25
        if c_end_cod >= n_cod:
            c_end_cod = n_cod - 1
            c_start_cod = c_end_cod - 25
        if c_start_cod < 1 or c_end_cod >= n_cod:
            return None
        sub = seq[c_start_cod * 3 : c_end_cod * 3]
        assert len(sub) == 75
        return sub[-510:]

    raise ValueError(f"Unknown condition: {condition}")


def predict_probs(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    kmer_seqs: list[str],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    probs_all: list[np.ndarray] = []
    with torch.inference_mode():
        for i in range(0, len(kmer_seqs), batch_size):
            batch = kmer_seqs[i : i + batch_size]
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
            probs_all.append(exps / np.sum(exps, axis=1, keepdims=True))
    return np.concatenate(probs_all, axis=0)


def compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    p1 = probs[:, 1]
    y_pred = (probs[:, 1] >= probs[:, 0]).astype(int)
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    fp = int(np.sum((y_pred == 1) & neg_mask))
    tn = int(np.sum((y_pred == 0) & neg_mask))
    return {
        "n_pos": int(np.sum(pos_mask)),
        "n_neg": int(np.sum(neg_mask)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "fpr": round(float(fp / (fp + tn)) if (fp + tn) else 0.0, 4),
        "auroc": round(float(roc_auc_score(y_true, p1)), 4) if len(set(y_true.tolist())) > 1 else 0.0,
        "mean_prob_pos": round(float(np.mean(p1[pos_mask])), 4),
        "mean_prob_neg": round(float(np.mean(p1[neg_mask])), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--piloto-dir", type=Path, default=DEFAULT_PILOTO)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.piloto_dir / "manifiesto_piloto.tsv").open(encoding="utf-8") as fh:
        manifest = list(csv.DictReader(fh, delimiter="\t"))

    pos_all: list[dict] = []
    neg_pool: list[dict] = []
    for row in manifest:
        acc = row["accession"]
        rec = next(SeqIO.parse(str(args.piloto_dir / "genomas" / f"{acc}.fna"), "fasta"))
        seq_u = str(rec.seq).upper()
        pos_rows = parse_pos_cds(args.piloto_dir / "genomas" / f"{acc}.gff", seq_u, rec.id)
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

    neg_sampled = sample_matched_negatives(pos_all, neg_pool, args.seed)
    dataset = pos_all + neg_sampled

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cds_dir = args.models_dir / "genelm_cds"
    tokenizer = AutoTokenizer.from_pretrained(str(cds_dir), local_files_only=True)
    model_cds = AutoModelForSequenceClassification.from_pretrained(
        str(cds_dir), local_files_only=True
    ).to(device).eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    conditions = (
        "complete_gt103",
        "sin_5prime_50pct",
        "sin_3prime_50pct_slice3p",
        "sin_3prime_50pct_slice5p",
        "sin_ambos_50pct",
        "truncado_corto_le100",
    )

    tsv_rows: list[dict] = []
    json_detail: dict = {
        "seed": args.seed,
        "pos_pool": len(pos_all),
        "neg_pool_available": len(neg_pool),
        "neg_sampled": len(neg_sampled),
        "pos_len_bins": dict(Counter(length_stratum(len(r["sequence"])) for r in pos_all)),
        "neg_sampled_len_bins": dict(Counter(length_stratum(len(r["sequence"])) for r in neg_sampled)),
        "conditions": [],
    }

    for cond in conditions:
        items = []
        kmers = []
        for r in dataset:
            sub = transform_sequence(r["sequence"], cond)
            if sub is None:
                continue
            items.append((r["label"], r["strand"]))
            kmers.append(generate_kmer(sub, k=6, overlap=3))

        probs = predict_probs(model_cds, tokenizer, kmers, device, args.batch_size)
        y_true = np.array([it[0] for it in items], dtype=int)
        strands = np.array([it[1] for it in items])

        overall = compute_metrics(y_true, probs)
        tsv_rows.append({"condition": cond, "strand": "ALL", **overall})
        cond_entry = {"condition": cond, "ALL": overall, "by_strand": {}}

        for st in ("+", "-"):
            mask = strands == st
            m_st = compute_metrics(y_true[mask], probs[mask])
            tsv_rows.append({"condition": cond, "strand": st, **m_st})
            cond_entry["by_strand"][st] = m_st

        json_detail["conditions"].append(cond_entry)
        print(
            f"[{cond:24s}] n_pos={overall['n_pos']:4d} n_neg={overall['n_neg']:4d} | "
            f"Acc={overall['accuracy']:.4f} Prec={overall['precision']:.4f} "
            f"Rec={overall['recall']:.4f} F1={overall['f1']:.4f} FPR={overall['fpr']:.4f} "
            f"AUROC={overall['auroc']:.4f}"
        )

    peak_mib = (
        round(torch.cuda.max_memory_allocated(device) / (1024**2), 2)
        if device.type == "cuda"
        else 0.0
    )
    json_detail["peak_vram_mib"] = peak_mib
    print(f"Peak VRAM: {peak_mib} MiB")

    tsv_path = args.out_dir / "resultados_truncamiento_directo.tsv"
    fields = list(tsv_rows[0].keys())
    with tsv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(tsv_rows)

    json_path = args.out_dir / "resultados_truncamiento_directo.json"
    json_path.write_text(json.dumps(json_detail, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
