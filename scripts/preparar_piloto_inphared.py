#!/usr/bin/env python3
"""Inspecciona el corpus INPHARED (GenomesDB_Jan_2026.tar.gz) en streaming y extrae
un piloto estratificado de 20 genomas de fagos completos para la Etapa 1 de GeneLM.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

_REPO_DIR = Path(__file__).resolve().parent.parent
_PARENT_DIR = _REPO_DIR.parent

from Bio.Seq import Seq

DEFAULT_TAR = (_PARENT_DIR / "trabajos_referencia/datasets/GenomesDB_Jan_2026.tar.gz"
)
DEFAULT_OUT = (_REPO_DIR / "etapa1_fagos_completos/piloto_inphared"
)

GENELM_ALLOWED = set("ACGTUN")
STANDARD_ACGTN = set("ACGTN")
START_CODONS_T11 = ("ATG", "GTG", "TTG", "ATT", "ATC", "ATA")
STOP_CODONS_T11 = ("TAA", "TAG", "TGA")


def parse_fasta_text(text: str) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header = None
    chunks: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks).upper()))
            header = line[1:].split()[0]
            chunks = []
        else:
            chunks.append(line)
    if header is not None:
        records.append((header, "".join(chunks).upper()))
    return records


def parse_gff_cds(text: str) -> list[dict]:
    cds_list: list[dict] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            if line == "##FASTA":
                break
            continue
        parts = line.split("\t")
        if len(parts) != 9:
            continue
        seqid, source, ftype, start_s, end_s, score, strand, phase, attrs = parts
        if ftype.lower() != "cds":
            continue
        cds_list.append(
            {
                "seqid": seqid,
                "source": source,
                "start_1": int(start_s),
                "end_1": int(end_s),
                "strand": strand,
                "phase": phase,
                "attrs": attrs,
                "has_phrog": ("phrog_" in attrs.lower()),
            }
        )
    return cds_list


def size_stratum(length_bp: int) -> str:
    if length_bp < 20_000:
        return "lt_20kb"
    if length_bp < 50_000:
        return "20kb_50kb"
    if length_bp < 100_000:
        return "50kb_100kb"
    return "ge_100kb"


def analyze_cds_codons(seq: str, cds_list: list[dict], record_id: str) -> dict:
    stats = {
        "n_cds": 0,
        "strand_pos": 0,
        "strand_neg": 0,
        "len_le_60": 0,
        "len_le_100": 0,
        "len_le_103": 0,
        "edge_lt30": 0,
        "has_phrog": 0,
        "mod3_ok": 0,
        "valid_stop": 0,
        "non_genelm_start": 0,
        "start_codons": Counter(),
        "stop_codons": Counter(),
        "seqid_mismatch": 0,
    }
    seq_len = len(seq)
    for cds in cds_list:
        if cds["seqid"] != record_id:
            stats["seqid_mismatch"] += 1
            continue
        s0 = cds["start_1"] - 1
        e0 = cds["end_1"]
        if s0 < 0 or e0 > seq_len or e0 <= s0:
            continue
        sub = seq[s0:e0]
        if cds["strand"] == "-":
            sub = str(Seq(sub).reverse_complement())
            stats["strand_neg"] += 1
            dist_5prime_edge = seq_len - e0
        elif cds["strand"] == "+":
            stats["strand_pos"] += 1
            dist_5prime_edge = s0
        else:
            continue

        stats["n_cds"] += 1
        cds_len = e0 - s0
        if cds_len <= 60:
            stats["len_le_60"] += 1
        if cds_len <= 100:
            stats["len_le_100"] += 1
        if cds_len <= 103:
            stats["len_le_103"] += 1
        if dist_5prime_edge < 30:
            stats["edge_lt30"] += 1
        if cds["has_phrog"]:
            stats["has_phrog"] += 1
        if cds_len % 3 == 0:
            stats["mod3_ok"] += 1

        sc = sub[:3]
        tc = sub[-3:]
        stats["start_codons"][sc] += 1
        stats["stop_codons"][tc] += 1
        if tc in STOP_CODONS_T11:
            stats["valid_stop"] += 1
        if sc not in ("ATG", "GTG", "TTG", "CTG"):
            stats["non_genelm_start"] += 1
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tar-path", type=Path, default=DEFAULT_TAR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_genomes = args.out_dir / "genomas"
    out_genomes.mkdir(parents=True, exist_ok=True)

    fna_members: dict[str, tarfile.TarInfo] = {}
    gff_members: dict[str, tarfile.TarInfo] = {}

    print(f"Escaneando {args.tar_path} en modo tar sin compresión...")
    with tarfile.open(args.tar_path, "r:") as tf:
        for member in tf:
            if not member.isfile():
                continue
            rel = member.name.lstrip("./")
            parts = rel.split("/")
            if len(parts) != 3 or parts[0] != "GenomesDB":
                continue
            acc, fname = parts[1], parts[2]
            if fname == f"{acc}.fna":
                fna_members[acc] = member
            elif fname == f"{acc}.gff":
                gff_members[acc] = member

        paired_accs = sorted(set(fna_members) & set(gff_members))
        print(f"Genomas con par .fna + .gff: {len(paired_accs)}")

        corpus_lengths: list[int] = []
        stratum_counts = Counter()
        multi_contig_accs: list[str] = []
        has_n_accs: list[str] = []
        iupac_non_acgtn_accs: dict[str, list[str]] = {}
        iupac_char_counts = Counter()
        gff_sources = Counter()
        global_cds = {
            "n_cds": 0,
            "strand_pos": 0,
            "strand_neg": 0,
            "len_le_60": 0,
            "len_le_100": 0,
            "len_le_103": 0,
            "edge_lt30": 0,
            "has_phrog": 0,
            "mod3_ok": 0,
            "valid_stop": 0,
            "non_genelm_start": 0,
            "seqid_mismatch": 0,
        }
        global_starts = Counter()
        global_stops = Counter()

        candidates_by_stratum: dict[str, list[dict]] = defaultdict(list)

        for acc in paired_accs:
            fna_bytes = tf.extractfile(fna_members[acc]).read()
            gff_bytes = tf.extractfile(gff_members[acc]).read()
            fna_text = fna_bytes.decode("utf-8", errors="replace")
            gff_text = gff_bytes.decode("utf-8", errors="replace")

            records = parse_fasta_text(fna_text)
            if len(records) != 1:
                multi_contig_accs.append(acc)
                continue

            record_id, seq = records[0]
            seq_len = len(seq)
            corpus_lengths.append(seq_len)
            st = size_stratum(seq_len)
            stratum_counts[st] += 1

            chars = set(seq)
            extra_iupac = sorted(chars - STANDARD_ACGTN)
            if "N" in chars:
                has_n_accs.append(acc)
            if extra_iupac:
                iupac_non_acgtn_accs[acc] = extra_iupac
                for ch in extra_iupac:
                    iupac_char_counts[ch] += 1

            cds_list = parse_gff_cds(gff_text)
            for c in cds_list:
                gff_sources[c["source"]] += 1

            c_stats = analyze_cds_codons(seq, cds_list, record_id)
            for k in global_cds:
                global_cds[k] += c_stats[k]
            global_starts.update(c_stats["start_codons"])
            global_stops.update(c_stats["stop_codons"])

            # Elegible para piloto limpio (solo ACGTN para correr los 3 modos sin error fatal)
            if not extra_iupac and c_stats["n_cds"] >= 5 and c_stats["seqid_mismatch"] == 0:
                candidates_by_stratum[st].append(
                    {
                        "accession": acc,
                        "record_id": record_id,
                        "length_bp": seq_len,
                        "stratum": st,
                        "has_n": int("N" in chars),
                        "n_cds": c_stats["n_cds"],
                        "strand_pos": c_stats["strand_pos"],
                        "strand_neg": c_stats["strand_neg"],
                        "len_le_100": c_stats["len_le_100"],
                        "len_le_103": c_stats["len_le_103"],
                        "edge_lt30": c_stats["edge_lt30"],
                        "has_phrog": c_stats["has_phrog"],
                        "valid_stop": c_stats["valid_stop"],
                        "non_genelm_start": c_stats["non_genelm_start"],
                    }
                )

        # Selección estratificada reproducible (seed=42)
        targets = {
            "lt_20kb": 4,
            "20kb_50kb": 6,
            "50kb_100kb": 6,
            "ge_100kb": 4,
        }
        rng = random.Random(args.seed)
        pilot_rows: list[dict] = []
        for st, k in targets.items():
            pool = sorted(candidates_by_stratum[st], key=lambda x: x["accession"])
            if len(pool) < k:
                raise RuntimeError(f"Estrato {st} tiene solo {len(pool)} candidatos (< {k})")
            chosen = rng.sample(pool, k)
            chosen.sort(key=lambda x: x["length_bp"])
            pilot_rows.extend(chosen)

        # Extraer los 20 pares .fna / .gff seleccionados
        for row in pilot_rows:
            acc = row["accession"]
            (out_genomes / f"{acc}.fna").write_bytes(
                tf.extractfile(fna_members[acc]).read()
            )
            (out_genomes / f"{acc}.gff").write_bytes(
                tf.extractfile(gff_members[acc]).read()
            )

    # Escribir manifiesto del piloto
    manifiesto_path = args.out_dir / "manifiesto_piloto.tsv"
    fields = [
        "accession",
        "record_id",
        "length_bp",
        "stratum",
        "has_n",
        "n_cds",
        "strand_pos",
        "strand_neg",
        "len_le_100",
        "len_le_103",
        "edge_lt30",
        "has_phrog",
        "valid_stop",
        "non_genelm_start",
    ]
    with manifiesto_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(pilot_rows)

    sorted_lens = sorted(corpus_lengths)
    n_mono = len(sorted_lens)
    corpus_summary = {
        "tar_path": str(args.tar_path),
        "paired_fna_gff_genomes": len(paired_accs),
        "mono_contig_genomes": n_mono,
        "multi_contig_genomes": len(multi_contig_accs),
        "multi_contig_accessions_sample": multi_contig_accs[:20],
        "genomes_with_N": len(has_n_accs),
        "genomes_with_non_acgtn_iupac": len(iupac_non_acgtn_accs),
        "non_acgtn_char_genome_counts": dict(iupac_char_counts.most_common()),
        "non_acgtn_accessions_sample": dict(list(iupac_non_acgtn_accs.items())[:25]),
        "stratum_genome_counts": dict(stratum_counts),
        "genome_length_bp_stats": {
            "min": sorted_lens[0],
            "p25": sorted_lens[n_mono // 4],
            "median": statistics.median(sorted_lens),
            "p75": sorted_lens[(3 * n_mono) // 4],
            "max": sorted_lens[-1],
        },
        "gff_feature_sources": dict(gff_sources.most_common()),
        "global_cds_stats": {
            **global_cds,
            "pct_len_le_60": round(100 * global_cds["len_le_60"] / global_cds["n_cds"], 3),
            "pct_len_le_100": round(100 * global_cds["len_le_100"] / global_cds["n_cds"], 3),
            "pct_len_le_103": round(100 * global_cds["len_le_103"] / global_cds["n_cds"], 3),
            "pct_edge_lt30": round(100 * global_cds["edge_lt30"] / global_cds["n_cds"], 3),
            "pct_has_phrog": round(100 * global_cds["has_phrog"] / global_cds["n_cds"], 3),
            "pct_mod3_ok": round(100 * global_cds["mod3_ok"] / global_cds["n_cds"], 3),
            "pct_valid_stop": round(100 * global_cds["valid_stop"] / global_cds["n_cds"], 3),
            "pct_non_genelm_start": round(100 * global_cds["non_genelm_start"] / global_cds["n_cds"], 3),
            "start_codons_top15": global_starts.most_common(15),
            "stop_codons_top10": global_stops.most_common(10),
        },
        "pilot_summary": {
            "seed": args.seed,
            "n_genomes": len(pilot_rows),
            "total_bp": sum(r["length_bp"] for r in pilot_rows),
            "total_cds": sum(r["n_cds"] for r in pilot_rows),
            "total_len_le_103": sum(r["len_le_103"] for r in pilot_rows),
            "total_edge_lt30": sum(r["edge_lt30"] for r in pilot_rows),
            "total_has_phrog": sum(r["has_phrog"] for r in pilot_rows),
            "total_valid_stop": sum(r["valid_stop"] for r in pilot_rows),
            "total_non_genelm_start": sum(r["non_genelm_start"] for r in pilot_rows),
        },
    }

    resumen_path = args.out_dir / "resumen_corpus_inphared.json"
    resumen_path.write_text(json.dumps(corpus_summary, indent=2) + "\n", encoding="utf-8")

    print(f"Manifiesto piloto escrito en: {manifiesto_path}")
    print(f"Resumen corpus escrito en: {resumen_path}")
    print("Resumen piloto:", json.dumps(corpus_summary["pilot_summary"], indent=2))
    print(
        "Corpus IUPAC no-ACGTN:",
        corpus_summary["genomes_with_non_acgtn_iupac"],
        "genomas | con N:",
        corpus_summary["genomes_with_N"],
    )
    print(
        "Corpus CDS <=103 nt:",
        f"{global_cds['len_le_103']}/{global_cds['n_cds']} ({corpus_summary['global_cds_stats']['pct_len_le_103']}%)",
        "| valid_stop:",
        f"{corpus_summary['global_cds_stats']['pct_valid_stop']}%",
        "| edge <30pb:",
        f"{corpus_summary['global_cds_stats']['pct_edge_lt30']}%",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
