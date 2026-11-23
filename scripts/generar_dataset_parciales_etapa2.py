#!/usr/bin/env python3
"""Genera el dataset estratificado de fragmentos tipo contig (1, 2, 5, 10 kb)
a partir de los 20 genomas piloto de INPHARED, proyectando etiquetas exactas
de truncamiento (complete, partial_5prime, partial_3prime, partial_both).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

_REPO_DIR = Path(__file__).resolve().parent.parent
_PARENT_DIR = _REPO_DIR.parent

from Bio import SeqIO
from Bio.Seq import Seq

DEFAULT_PILOTO = (_REPO_DIR / "etapa1_fagos_completos/piloto_inphared"
)
DEFAULT_OUT = (_REPO_DIR / "etapa2_datos_parciales/dataset_fragmentos"
)

WINDOW_CAPS = {
    1_000: 45,
    2_000: 18,
    5_000: 10,
    10_000: 8,
}
VALID_STARTS = {"ATG", "GTG", "TTG"}
VALID_STOPS = {"TAA", "TAG", "TGA"}


def stable_rng(seed: int, record_id: str, length: int) -> random.Random:
    key = f"{seed}|{record_id}|{length}|1".encode("utf-8")
    derived = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    return random.Random(derived)


def parse_gff_cds(gff_path: Path, record_id: str) -> list[dict]:
    cds_rows: list[dict] = []
    for line_no, raw in enumerate(gff_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            if line == "##FASTA":
                break
            continue
        parts = line.split("\t")
        if len(parts) != 9:
            continue
        seqid, _src, ftype, s1_s, e1_s, _sc, strand_s, _ph, attrs = parts
        if ftype.lower() != "cds":
            continue
        if seqid != record_id:
            raise ValueError(f"{gff_path}:{line_no}: seqid {seqid!r} != {record_id!r}")
        if strand_s not in {"+", "-"}:
            continue
        attr_map = {}
        for item in attrs.split(";"):
            if "=" in item:
                k, v = item.split("=", 1)
                attr_map[k] = v
        gene_id = attr_map.get("ID") or attr_map.get("locus_tag") or f"{record_id}_cds_{line_no}"
        s0 = int(s1_s) - 1
        e0 = int(e1_s)
        cds_rows.append(
            {
                "record_id": record_id,
                "gene_id": gene_id,
                "gene_start_0": s0,
                "gene_end_0": e0,
                "gene_length_bp": e0 - s0,
                "strand": 1 if strand_s == "+" else -1,
                "has_phrog": int("phrog_" in attrs.lower()),
                "product": attr_map.get("product", ""),
            }
        )
    return cds_rows


def sample_windows(record_id: str, genome_len: int, seed: int) -> list[dict]:
    windows: list[dict] = []
    for length, cap in WINDOW_CAPS.items():
        if genome_len < length:
            continue
        possible_starts = genome_len - length + 1
        n_win = min(math.ceil(genome_len / length), cap, possible_starts)
        rng = stable_rng(seed, record_id, length)
        for stratum in range(n_win):
            lo = stratum * possible_starts // n_win
            hi = (stratum + 1) * possible_starts // n_win
            start = rng.randrange(lo, hi)
            windows.append(
                {
                    "fragment_id": f"{record_id}|L{length}|w{stratum + 1:02d}",
                    "record_id": record_id,
                    "length": length,
                    "stratum_idx": stratum + 1,
                    "seed": seed,
                    "start": start,
                    "end": start + length,
                }
            )
    return windows


def project_cds_and_verify(window: dict, frag_seq: str, cds_list: list[dict]) -> list[dict]:
    a, b, win_len = window["start"], window["end"], window["length"]
    projected: list[dict] = []
    for gene in cds_list:
        g_s, g_e, strand = gene["gene_start_0"], gene["gene_end_0"], gene["strand"]
        vis_s = max(a, g_s)
        vis_e = min(b, g_e)
        if vis_s >= vis_e:
            continue
        loc_s = vis_s - a
        loc_e = vis_e - a
        obs_bp = loc_e - loc_s

        if strand == 1:
            partial_5prime = a > g_s
            partial_3prime = b < g_e
        else:
            partial_5prime = b < g_e
            partial_3prime = a > g_s

        if partial_5prime and partial_3prime:
            status = "partial_both"
        elif partial_5prime:
            status = "partial_5prime"
        elif partial_3prime:
            status = "partial_3prime"
        else:
            status = "complete"

        tis_observable = int((not partial_5prime) and obs_bp >= 3)

        # Verificación obligatoria de bordes físicos y codones en el fragmento extraído
        sub = frag_seq[loc_s:loc_e]
        if strand == -1:
            sub = str(Seq(sub).reverse_complement())

        if status == "complete":
            assert loc_s >= 0 and loc_e <= win_len
            assert sub[:3] in VALID_STARTS, f"complete bad start {sub[:3]} en {window['fragment_id']} {gene['gene_id']}"
            assert sub[-3:] in VALID_STOPS, f"complete bad stop {sub[-3:]} en {window['fragment_id']} {gene['gene_id']}"
        elif status == "partial_5prime":
            if strand == 1:
                assert loc_s == 0
            else:
                assert loc_e == win_len
            if obs_bp >= 3:
                assert sub[-3:] in VALID_STOPS, f"partial_5prime bad stop {sub[-3:]} en {window['fragment_id']}"
        elif status == "partial_3prime":
            if strand == 1:
                assert loc_e == win_len
            else:
                assert loc_s == 0
            if obs_bp >= 3:
                assert sub[:3] in VALID_STARTS, f"partial_3prime bad start {sub[:3]} en {window['fragment_id']}"
        elif status == "partial_both":
            assert loc_s == 0 and loc_e == win_len

        projected.append(
            {
                "record_id": gene["record_id"],
                "fragment_id": window["fragment_id"],
                "fragment_length": win_len,
                "gene_id": gene["gene_id"],
                "gene_start_0": g_s,
                "gene_end_0": g_e,
                "gene_length_bp": gene["gene_length_bp"],
                "visible_start_local_0": loc_s,
                "visible_end_local_0": loc_e,
                "strand": strand,
                "observable_bp": obs_bp,
                "truncation_status": status,
                "tis_observable": tis_observable,
                "has_phrog": gene["has_phrog"],
                "product": gene["product"],
            }
        )
    return projected


def write_tsv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--piloto-dir", type=Path, default=DEFAULT_PILOTO)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.piloto_dir / "manifiesto_piloto.tsv").open(encoding="utf-8") as fh:
        manifest = list(csv.DictReader(fh, delimiter="\t"))

    windows_all: list[dict] = []
    refs_all: list[dict] = []
    by_len_status: dict[str, Counter] = defaultdict(Counter)
    status_counts = Counter()
    gt103_counts = Counter()
    le103_counts = Counter()
    len_counts = Counter()

    fna_out = args.out_dir / "fragmentos.fna"
    with fna_out.open("w", encoding="utf-8") as fasta_fh:
        for row in manifest:
            acc = row["accession"]
            rec = next(SeqIO.parse(str(args.piloto_dir / "genomas" / f"{acc}.fna"), "fasta"))
            seq_u = str(rec.seq).upper()
            cds_list = parse_gff_cds(args.piloto_dir / "genomas" / f"{acc}.gff", rec.id)
            wins = sample_windows(rec.id, len(seq_u), args.seed)
            windows_all.extend(wins)
            for w in wins:
                frag_seq = seq_u[w["start"] : w["end"]]
                len_counts[w["length"]] += 1
                fasta_fh.write(f">{w['fragment_id']}\n")
                for pos in range(0, len(frag_seq), 80):
                    fasta_fh.write(frag_seq[pos : pos + 80] + "\n")
                proj = project_cds_and_verify(w, frag_seq, cds_list)
                refs_all.extend(proj)
                for p in proj:
                    st = p["truncation_status"]
                    status_counts[st] += 1
                    by_len_status[str(w["length"])][st] += 1
                    if p["observable_bp"] > 103:
                        gt103_counts[st] += 1
                    else:
                        le103_counts[st] += 1

    win_fields = ["fragment_id", "record_id", "length", "stratum_idx", "seed", "start", "end"]
    ref_fields = [
        "record_id",
        "fragment_id",
        "fragment_length",
        "gene_id",
        "gene_start_0",
        "gene_end_0",
        "gene_length_bp",
        "visible_start_local_0",
        "visible_end_local_0",
        "strand",
        "observable_bp",
        "truncation_status",
        "tis_observable",
        "has_phrog",
        "product",
    ]
    write_tsv(args.out_dir / "ventanas.tsv", windows_all, win_fields)
    write_tsv(args.out_dir / "referencias_cds_observables.tsv", refs_all, ref_fields)

    summary = {
        "seed": args.seed,
        "genomes": len(manifest),
        "total_fragments": len(windows_all),
        "total_fragment_bp": sum(w["length"] for w in windows_all),
        "fragments_by_length": {str(k): len_counts[k] for k in sorted(len_counts)},
        "total_observable_cds": len(refs_all),
        "counts_by_status": dict(status_counts),
        "counts_by_status_gt103": dict(gt103_counts),
        "counts_by_status_le103": dict(le103_counts),
        "counts_by_length_and_status": {k: dict(v) for k, v in sorted(by_len_status.items(), key=lambda x: int(x[0]))},
    }
    (args.out_dir / "resumen_fragmentos.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Fragmentos: {len(windows_all)} | Total bp: {summary['total_fragment_bp']:,} | CDS observables: {len(refs_all)}")
    print("Por longitud:", summary["fragments_by_length"])
    print("Por categoría:", summary["counts_by_status"])
    print("Subconjunto >103 nt:", summary["counts_by_status_gt103"])
    print("Subconjunto <=103 nt:", summary["counts_by_status_le103"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
