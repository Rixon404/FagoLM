#!/usr/bin/env python3
"""Paso 2 (Etapa 3): Construye el split libre de fuga por genoma y homología (21-mers < 0.05)
en Train (120 fagos) y Val (20 fagos) respecto a Test (20 fagos congelados), genera fragmentos
y construye los datasets balanceados para FagoLM-CDS y FagoLM-TIS.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import tarfile
import zlib
from collections import Counter, defaultdict
from pathlib import Path

_REPO_DIR = Path(__file__).resolve().parent.parent
_PARENT_DIR = _REPO_DIR.parent

from Bio import SeqIO
from Bio.Seq import Seq

DEFAULT_TAR = (_PARENT_DIR / "trabajos_referencia/datasets/GenomesDB_Jan_2026.tar.gz"
)
DEFAULT_TEST_DIR = (_REPO_DIR / "etapa1_fagos_completos/piloto_inphared"
)
DEFAULT_STAGE3_DIR = (_REPO_DIR / "etapa3_finetuning_lora"
)

TRAIN_TARGETS = {"lt_20kb": 24, "20kb_50kb": 36, "50kb_100kb": 36, "ge_100kb": 24}
VAL_TARGETS = {"lt_20kb": 4, "20kb_50kb": 6, "50kb_100kb": 6, "ge_100kb": 4}

TRAIN_WINDOW_CAPS = {1_000: 25, 2_000: 14, 5_000: 7, 10_000: 4}
VAL_WINDOW_CAPS = {1_000: 45, 2_000: 18, 5_000: 10, 10_000: 8}

STARTS_CORE = ("TTG", "CTG", "ATG", "GTG")
STOPS_CORE = ("TAA", "TAG", "TGA")
ALLOWED_ACGTN = set("ACGTN")


def size_stratum(length_bp: int) -> str:
    if length_bp < 20_000:
        return "lt_20kb"
    if length_bp < 50_000:
        return "20kb_50kb"
    if length_bp < 100_000:
        return "50kb_100kb"
    return "ge_100kb"


def parse_fasta_text(text: str) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    hdr = None
    chunks: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if hdr is not None:
                records.append((hdr, "".join(chunks).upper()))
            hdr = line[1:].split()[0]
            chunks = []
        else:
            chunks.append(line)
    if hdr is not None:
        records.append((hdr, "".join(chunks).upper()))
    return records


def parse_gff_cds(text: str, record_id: str) -> list[dict]:
    cds: list[dict] = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            if line == "##FASTA":
                break
            continue
        parts = line.split("\t")
        if len(parts) != 9 or parts[2].lower() != "cds":
            continue
        seqid, _src, _ft, s1_s, e1_s, _sc, strand_s, _ph, attrs = parts
        if seqid != record_id or strand_s not in {"+", "-"}:
            continue
        attr_map = dict(item.split("=", 1) for item in attrs.split(";") if "=" in item)
        cds.append(
            {
                "record_id": record_id,
                "gene_id": attr_map.get("ID") or attr_map.get("locus_tag") or f"{record_id}_cds_{line_no}",
                "gene_start_0": int(s1_s) - 1,
                "gene_end_0": int(e1_s),
                "gene_length_bp": int(e1_s) - (int(s1_s) - 1),
                "strand": 1 if strand_s == "+" else -1,
                "has_phrog": int("phrog_" in attrs.lower()),
            }
        )
    return cds


def kmer21_sketch(seq_u: str) -> set[bytes]:
    """Sketch determinístico de ~1/16 de los 21-mers canónicos sin 'N'."""
    seq_rc = str(Seq(seq_u).reverse_complement())
    L = len(seq_u)
    sketch: set[bytes] = set()
    for i in range(0, L - 20):
        km = seq_u[i : i + 21]
        if "N" in km:
            continue
        km_rc = seq_rc[L - i - 21 : L - i]
        canon = km if km < km_rc else km_rc
        b = canon.encode("ascii")
        if (zlib.crc32(b) & 15) == 0:
            sketch.add(b)
    return sketch


def containment(a: set[bytes], b: set[bytes]) -> float:
    denom = min(len(a), len(b))
    if denom == 0:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    inter = sum(1 for x in a if x in b)
    return inter / denom


def stable_rng(seed: int, record_id: str, length: int) -> random.Random:
    key = f"{seed}|{record_id}|{length}|1".encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))


def sample_windows(record_id: str, genome_len: int, caps: dict[int, int], seed: int) -> list[dict]:
    windows: list[dict] = []
    for length, cap in caps.items():
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


def project_cds(window: dict, cds_list: list[dict]) -> list[dict]:
    a, b, win_len = window["start"], window["end"], window["length"]
    out: list[dict] = []
    for gene in cds_list:
        g_s, g_e, strand = gene["gene_start_0"], gene["gene_end_0"], gene["strand"]
        vis_s, vis_e = max(a, g_s), min(b, g_e)
        if vis_s >= vis_e:
            continue
        loc_s, loc_e = vis_s - a, vis_e - a
        obs_bp = loc_e - loc_s
        if strand == 1:
            p5 = a > g_s
            p3 = b < g_e
        else:
            p5 = b < g_e
            p3 = a > g_s
        if p5 and p3:
            status = "partial_both"
        elif p5:
            status = "partial_5prime"
        elif p3:
            status = "partial_3prime"
        else:
            status = "complete"
        out.append(
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
                "strand_sym": "+" if strand == 1 else "-",
                "observable_bp": obs_bp,
                "truncation_status": status,
                "tis_observable": int((not p5) and obs_bp >= 3),
                "has_phrog": gene["has_phrog"],
            }
        )
    return out


def interval_iou(a0: int, a1: int, b0: int, b1: int) -> float:
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union else 0.0


def oriented_to_genomic(strand: str, o_start: int, o_end: int, L: int) -> tuple[int, int]:
    if strand == "+":
        return o_start, o_end
    return L - o_end, L - o_start


def genomic_to_oriented(strand: str, g_start: int, g_end: int, L: int) -> tuple[int, int]:
    if strand == "+":
        return g_start, g_end
    return L - g_end, L - g_start


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


def extract_open_groups_ge60(frag_id: str, seq_u: str, seq_rc: str) -> list[dict]:
    """Extrae grupos abiertos (complete, open_5, open_3, open_both) con in_frame_len >= 60 nt."""
    L = len(seq_u)
    starts_set = set(STARTS_CORE)
    stops_set = set(STOPS_CORE)
    groups: list[dict] = []

    for strand, oriented in (("+", seq_u), ("-", seq_rc)):
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
                    gs, ge = oriented_to_genomic(strand, 0, first_stop_end, L)
                    groups.append(
                        {
                            "fragment_id": frag_id,
                            "strand": strand,
                            "frame": frame,
                            "region_kind": "open_5",
                            "g_start0": gs,
                            "g_end0": ge,
                            "o_start": 0,
                            "o_end": first_stop_end,
                            "in_frame_len": len(in_frame_seq),
                            "sequence_510": in_frame_seq[-510:],
                        }
                    )

                # 2) complete (stops posteriores)
                for k_idx in range(1, len(stop_indices)):
                    prev_si = stop_indices[k_idx - 1]
                    cur_si = stop_indices[k_idx]
                    stop_end = codon_starts[cur_si] + 3
                    internal = [si for si in start_indices if prev_si < si < cur_si]
                    if not internal:
                        continue
                    first_st_pos = codon_starts[internal[0]]
                    orf_seq = oriented[first_st_pos:stop_end]
                    if len(orf_seq) >= 60:
                        gs, ge = oriented_to_genomic(strand, first_st_pos, stop_end, L)
                        groups.append(
                            {
                                "fragment_id": frag_id,
                                "strand": strand,
                                "frame": frame,
                                "region_kind": "complete",
                                "g_start0": gs,
                                "g_end0": ge,
                                "o_start": first_st_pos,
                                "o_end": stop_end,
                                "in_frame_len": len(orf_seq),
                                "sequence_510": orf_seq[-510:],
                            }
                        )

                # 3) open_3 (tras último stop)
                last_si = stop_indices[-1]
                after = [si for si in start_indices if si > last_si]
                if after:
                    first_st_pos = codon_starts[after[0]]
                    in_frame_seq = oriented[first_st_pos:aligned_end]
                    if len(in_frame_seq) >= 60:
                        gs, ge = oriented_to_genomic(strand, first_st_pos, L, L)
                        groups.append(
                            {
                                "fragment_id": frag_id,
                                "strand": strand,
                                "frame": frame,
                                "region_kind": "open_3",
                                "g_start0": gs,
                                "g_end0": ge,
                                "o_start": first_st_pos,
                                "o_end": L,
                                "in_frame_len": len(in_frame_seq),
                                "sequence_510": in_frame_seq[-510:],
                            }
                        )
            else:
                # 4) open_both (sin stops en todo el marco)
                in_frame_seq = oriented[frame:aligned_end]
                if len(in_frame_seq) >= 60:
                    gs, ge = oriented_to_genomic(strand, 0, L, L)
                    groups.append(
                        {
                            "fragment_id": frag_id,
                            "strand": strand,
                            "frame": frame,
                            "region_kind": "open_both",
                            "g_start0": gs,
                            "g_end0": ge,
                            "o_start": 0,
                            "o_end": L,
                            "in_frame_len": len(in_frame_seq),
                            "sequence_510": in_frame_seq[-510:],
                        }
                    )
    return groups


def extract_tis_examples_from_fragment(
    frag_seq_u: str, frag_seq_rc: str, proj_refs: list[dict], record_id: str
) -> tuple[list[dict], list[dict]]:
    """Extrae positivos (start verdadero en complete/partial_3prime) y negativos duros
    (inicios alternativos en complete/partial_3prime e inicios internos dentro de partial_5prime/partial_both).
    """
    L = len(frag_seq_u)
    starts_set = set(STARTS_CORE)
    pos_rows: list[dict] = []
    neg_rows: list[dict] = []

    for r in proj_refs:
        strand = r["strand_sym"]
        oriented = frag_seq_u if strand == "+" else frag_seq_rc
        g_s, g_e = r["visible_start_local_0"], r["visible_end_local_0"]
        o_s, o_e = genomic_to_oriented(strand, g_s, g_e, L)
        status = r["truncation_status"]

        if r["tis_observable"] == 1:
            # Positivo verdadero si tiene al menos 12 nt en marco hasta el borde/stop
            if o_e - o_s >= 12:
                win_pos = extract_tis_window_safe(frag_seq_u, frag_seq_rc, strand, g_s, g_e, w=30)
                if len(win_pos) == 60:
                    pos_rows.append(
                        {
                            "sequence": win_pos,
                            "label": 1,
                            "source_type": f"pos_{status}",
                            "record_id": record_id,
                        }
                    )
            # Negativos: codones ATG/GTG/TTG/CTG internos en el MISMO marco de lectura
            for pos in range(o_s + 3, o_e - 11, 3):
                if oriented[pos : pos + 3] in starts_set:
                    gs_n, ge_n = oriented_to_genomic(strand, pos, o_e, L)
                    win_n = extract_tis_window_safe(frag_seq_u, frag_seq_rc, strand, gs_n, ge_n, w=30)
                    if len(win_n) == 60:
                        neg_rows.append(
                            {
                                "sequence": win_n,
                                "label": 0,
                                "source_type": f"neg_alt_{status}",
                                "record_id": record_id,
                            }
                        )
        else:
            # partial_5prime o partial_both: alinear al marco real de la CDS usando el extremo 3' (en partial_5prime)
            # o usando frame del gen original proyectado (en partial_both)
            if status == "partial_5prime":
                # El extremo o_e termina exactamente en el stop codon -> marco = (o_e % 3)
                frame = o_e % 3
                start_scan = o_s + ((frame - (o_s % 3)) % 3)
            else:
                # partial_both: desplazar desde g_start_0 / g_end_0 del gen completo
                if strand == "+":
                    # window start a = g_s - 0? No: tenemos gene_start_0 y visible_start_local_0 == 0
                    # En realidad en +: fragment_start_genomic + o_s = window_start; fase respecto a gene_start_0:
                    # usamos el residuo de longitud consumida antes del borde 5'
                    # Para +: cut_5 = gene_length_bp - (L_visible + cut_3) -> mejor simplemente escanear los 3? No!
                    # Podemos calcular exacto: o_s == 0, cuánto se cortó en 5' no está aquí, pero si el gen es múltiplo de 3
                    # en +, el stop está a g_end_0, luego el marco relativo al fragmento es (g_end_0 - window_start) % 3,
                    # o equivalentemente: necesitamos el frame en el fragmento -> guardemos frame_oriented!
                    start_scan = o_s
                else:
                    start_scan = o_s
            # Para partial_both también sabemos el marco exacto calculando desde gene_start_0 en project_cds si lo guardamos;
            # pero añadamos frame_oriented exacto antes de entrar aquí.
            if status == "partial_both":
                start_scan = o_s + r["prefix_offset_in_frame"]

            for pos in range(start_scan, o_e - 11, 3):
                if oriented[pos : pos + 3] in starts_set:
                    gs_n, ge_n = oriented_to_genomic(strand, pos, o_e, L)
                    win_n = extract_tis_window_safe(frag_seq_u, frag_seq_rc, strand, gs_n, ge_n, w=30)
                    if len(win_n) == 60:
                        neg_rows.append(
                            {
                                "sequence": win_n,
                                "label": 0,
                                "source_type": f"neg_open5_{status}",
                                "record_id": record_id,
                            }
                        )
    return pos_rows, neg_rows


def build_cds_examples_from_genomes(
    genomes_data: list[dict],
    caps: dict[int, int],
    seed: int,
    save_fragments_dir: Path | None = None,
    supplement_short: bool = False,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Genera fragmentos, candidatos CDS y ejemplos TIS para una partición (Train o Val)."""
    rng = random.Random(seed)
    cds_pos: list[dict] = []
    cds_neg: list[dict] = []
    tis_pos: list[dict] = []
    tis_neg: list[dict] = []

    windows_all: list[dict] = []
    refs_all: list[dict] = []
    fna_lines: list[str] = []

    for g in genomes_data:
        acc = g["accession"]
        seq_u = g["seq"]
        cds_list = g["cds"]
        wins = sample_windows(acc, len(seq_u), caps, seed)
        for w in wins:
            frag_seq = seq_u[w["start"] : w["end"]]
            frag_rc = str(Seq(frag_seq).reverse_complement())
            proj = project_cds(w, cds_list)

            # Añadir offset exacto de marco para partial_both (para alinear el scan TIS en frame real)
            a, b = w["start"], w["end"]
            for r in proj:
                if r["strand_sym"] == "+":
                    # En +, inicio real del gen es gene_start_0; dentro del fragmento [a, b),
                    # posición en marco desde a es (gene_start_0 - a) % 3
                    r["prefix_offset_in_frame"] = (r["gene_start_0"] - a) % 3
                else:
                    # En -, oriented 0 corresponde a b; fin real del gen en + es gene_end_0,
                    # así que en oriented el inicio real está en b - gene_end_0, módulo 3:
                    r["prefix_offset_in_frame"] = (b - r["gene_end_0"]) % 3

            if save_fragments_dir is not None:
                windows_all.append(w)
                refs_all.extend(proj)
                fna_lines.append(f">{w['fragment_id']}\n")
                for p in range(0, len(frag_seq), 80):
                    fna_lines.append(frag_seq[p : p + 80] + "\n")

            # Indexar referencias por hebra
            refs_by_strand: dict[str, list[dict]] = {"+": [], "-": []}
            L = len(frag_seq)
            for r in proj:
                ro_s, ro_e = genomic_to_oriented(r["strand_sym"], r["visible_start_local_0"], r["visible_end_local_0"], L)
                r["o_start"] = ro_s
                r["o_end"] = ro_e
                if r["strand_sym"] == "+":
                    r["o_frame"] = (r["gene_start_0"] - a) % 3
                else:
                    r["o_frame"] = (b - r["gene_end_0"]) % 3
                refs_by_strand[r["strand_sym"]].append(r)

            groups = extract_open_groups_ge60(w["fragment_id"], frag_seq, frag_rc)
            for grp in groups:
                strand = grp["strand"]
                best_iou = 0.0
                best_ref = None
                for r in refs_by_strand[strand]:
                    iou = interval_iou(grp["g_start0"], grp["g_end0"], r["visible_start_local_0"], r["visible_end_local_0"])
                    if iou > best_iou:
                        best_iou = iou
                        best_ref = r

                len_bucket = "60_103" if grp["in_frame_len"] <= 103 else "gt103"
                if best_ref is not None and best_iou >= 0.5 and grp["frame"] == best_ref["o_frame"]:
                    cds_pos.append(
                        {
                            "sequence": grp["sequence_510"],
                            "label": 1,
                            "category": f"{grp['region_kind']}__{best_ref['truncation_status']}",
                            "region_kind": grp["region_kind"],
                            "len_bucket": len_bucket,
                            "record_id": acc,
                        }
                    )
                elif best_iou < 0.2:
                    cds_neg.append(
                        {
                            "sequence": grp["sequence_510"],
                            "label": 0,
                            "category": f"neg_{grp['region_kind']}",
                            "region_kind": grp["region_kind"],
                            "len_bucket": len_bucket,
                            "record_id": acc,
                        }
                    )

            tp, tn = extract_tis_examples_from_fragment(frag_seq, frag_rc, proj, acc)
            tis_pos.extend(tp)
            tis_neg.extend(tn)

        # Suplemento sintético controlado exclusivamente desde CDS reales de Train (para 60-103 nt y open_both)
        if supplement_short:
            for gene in cds_list:
                gs, ge = gene["gene_start_0"], gene["gene_end_0"]
                sub = seq_u[gs:ge]
                if gene["strand"] == -1:
                    sub = str(Seq(sub).reverse_complement())
                L_g = len(sub)
                if L_g % 3 != 0:
                    continue
                # 1) open_5 corto (60-102 nt terminando en stop)
                if L_g >= 120:
                    k_len = rng.choice([60, 69, 78, 87, 96, 102])
                    cds_pos.append(
                        {
                            "sequence": sub[-k_len:],
                            "label": 1,
                            "category": "open_5__partial_5prime_supp",
                            "region_kind": "open_5",
                            "len_bucket": "60_103",
                            "record_id": acc,
                        }
                    )
                    # 2) open_3 corto (60-102 nt empezando en inicio, sin stop)
                    k_len3 = rng.choice([60, 69, 78, 87, 96, 102])
                    cds_pos.append(
                        {
                            "sequence": sub[:k_len3],
                            "label": 1,
                            "category": "open_3__partial_3prime_supp",
                            "region_kind": "open_3",
                            "len_bucket": "60_103",
                            "record_id": acc,
                        }
                    )
                # 3) open_both interno (tanto 60-103 nt como >103 nt hasta 510 nt, sin inicio ni stop)
                if L_g >= 180:
                    n_cod = L_g // 3
                    mid_start = rng.randrange(1, max(2, n_cod // 2)) * 3
                    max_take = min(510, L_g - 3 - mid_start)
                    if max_take >= 60:
                        k_len_b = rng.choice([66, 78, 90, 99, min(300, max_take), max_take])
                        k_len_b = (k_len_b // 3) * 3
                        if 60 <= k_len_b <= max_take:
                            cds_pos.append(
                                {
                                    "sequence": sub[mid_start : mid_start + k_len_b],
                                    "label": 1,
                                    "category": "open_both__partial_both_supp",
                                    "region_kind": "open_both",
                                    "len_bucket": "60_103" if k_len_b <= 103 else "gt103",
                                    "record_id": acc,
                                }
                            )

    if save_fragments_dir is not None:
        save_fragments_dir.mkdir(parents=True, exist_ok=True)
        (save_fragments_dir / "fragmentos.fna").write_text("".join(fna_lines), encoding="utf-8")
        win_fields = ["fragment_id", "record_id", "length", "stratum_idx", "seed", "start", "end"]
        with (save_fragments_dir / "ventanas.tsv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=win_fields, delimiter="\t")
            w.writeheader()
            w.writerows(windows_all)
        ref_fields = [
            "record_id", "fragment_id", "fragment_length", "gene_id", "gene_start_0",
            "gene_end_0", "gene_length_bp", "visible_start_local_0", "visible_end_local_0",
            "strand", "observable_bp", "truncation_status", "tis_observable", "has_phrog",
        ]
        with (save_fragments_dir / "referencias_cds_observables.tsv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=ref_fields, delimiter="\t", extrasaction="ignore")
            w.writeheader()
            w.writerows(refs_all)

    return cds_pos, cds_neg, tis_pos, tis_neg


def sample_balanced_cds(
    pos_pool: list[dict], neg_pool: list[dict], total_target: int, seed: int, allow_supp_neg_both: bool = True
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    half = total_target // 2

    # Cuotas deseadas en positivos para cubrir todos los regímenes (completo, abiertos y cortos <=103 nt)
    quotas_pos = {
        ("complete", "gt103"): int(half * 0.25),
        ("complete", "60_103"): int(half * 0.11),
        ("open_5", "gt103"): int(half * 0.16),
        ("open_5", "60_103"): int(half * 0.09),
        ("open_3", "gt103"): int(half * 0.16),
        ("open_3", "60_103"): int(half * 0.09),
        ("open_both", "gt103"): int(half * 0.09),
        ("open_both", "60_103"): int(half * 0.05),
    }

    pos_by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in pos_pool:
        pos_by_key[(r["region_kind"], r["len_bucket"])].append(r)
    for k in pos_by_key:
        rng.shuffle(pos_by_key[k])

    chosen_pos: list[dict] = []
    used_pos_ids: set[int] = set()
    for k, q in quotas_pos.items():
        pool_k = pos_by_key[k]
        take = min(q, len(pool_k))
        for item in pool_k[:take]:
            chosen_pos.append(item)
            used_pos_ids.add(id(item))

    if len(chosen_pos) < half:
        leftover = [r for r in pos_pool if id(r) not in used_pos_ids]
        rng.shuffle(leftover)
        chosen_pos.extend(leftover[: half - len(chosen_pos)])

    # Emparejar negativos en exacta proporción (region_kind, len_bucket) de los positivos elegidos
    pos_counts = Counter((r["region_kind"], r["len_bucket"]) for r in chosen_pos)
    neg_by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in neg_pool:
        neg_by_key[(r["region_kind"], r["len_bucket"])].append(r)
    for k in neg_by_key:
        rng.shuffle(neg_by_key[k])

    # Si falta neg open_both corto/largo, recortar segmentos internos en marco de negativos open_5/open_3/complete
    if allow_supp_neg_both:
        for lb in ("60_103", "gt103"):
            needed_nb = pos_counts[("open_both", lb)] - len(neg_by_key[("open_both", lb)])
            if needed_nb > 0:
                donors = [
                    r for r in neg_pool
                    if r["region_kind"] in ("complete", "open_5", "open_3") and len(r["sequence"]) >= 120
                ]
                rng.shuffle(donors)
                for d in donors[:needed_nb]:
                    s = d["sequence"]
                    k_len = rng.choice([66, 78, 90, 99]) if lb == "60_103" else min(len(s) - 6, 300)
                    k_len = (k_len // 3) * 3
                    if k_len >= 60 and len(s) >= k_len + 6:
                        neg_by_key[("open_both", lb)].append(
                            {
                                "sequence": s[3 : 3 + k_len],
                                "label": 0,
                                "category": "neg_open_both_supp",
                                "region_kind": "open_both",
                                "len_bucket": lb,
                                "record_id": d["record_id"],
                            }
                        )

    chosen_neg: list[dict] = []
    used_neg_ids: set[int] = set()
    for k, q in pos_counts.items():
        pool_k = neg_by_key[k]
        take = min(q, len(pool_k))
        for item in pool_k[:take]:
            chosen_neg.append(item)
            used_neg_ids.add(id(item))

    # Completar manteniendo prioridad exacta de proporción 60_103 vs gt103
    if len(chosen_neg) < len(chosen_pos):
        target_short = sum(1 for r in chosen_pos if r["len_bucket"] == "60_103")
        cur_short = sum(1 for r in chosen_neg if r["len_bucket"] == "60_103")
        need_short = max(0, target_short - cur_short)
        rem_short = [r for r in neg_pool if id(r) not in used_neg_ids and r["len_bucket"] == "60_103"]
        rng.shuffle(rem_short)
        for item in rem_short[:need_short]:
            chosen_neg.append(item)
            used_neg_ids.add(id(item))

        rem_all = [r for r in neg_pool if id(r) not in used_neg_ids]
        rng.shuffle(rem_all)
        chosen_neg.extend(rem_all[: len(chosen_pos) - len(chosen_neg)])

    n_final = min(len(chosen_pos), len(chosen_neg))
    rows = chosen_pos[:n_final] + chosen_neg[:n_final]
    rng.shuffle(rows)
    stats = {
        "total": len(rows),
        "pos": n_final,
        "neg": n_final,
        "pos_by_bucket": dict(Counter(r["len_bucket"] for r in chosen_pos[:n_final])),
        "neg_by_bucket": dict(Counter(r["len_bucket"] for r in chosen_neg[:n_final])),
        "pos_by_region": dict(Counter(r["region_kind"] for r in chosen_pos[:n_final])),
        "neg_by_region": dict(Counter(r["region_kind"] for r in chosen_neg[:n_final])),
    }
    return rows, stats


def sample_balanced_tis(
    pos_pool: list[dict], neg_pool: list[dict], total_target: int, seed: int
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    pos_shuf = list(pos_pool)
    rng.shuffle(pos_shuf)

    neg_alt = [r for r in neg_pool if r["source_type"].startswith("neg_alt_")]
    neg_open5 = [r for r in neg_pool if r["source_type"].startswith("neg_open5_")]
    rng.shuffle(neg_alt)
    rng.shuffle(neg_open5)

    half = min(total_target // 2, len(pos_shuf), len(neg_pool))
    chosen_pos = pos_shuf[:half]

    # Mitad de negativos = inicios alternativos falsos; mitad = inicios internos en partial_5/partial_both
    n_alt = min(half // 2, len(neg_alt))
    n_open5 = min(half - n_alt, len(neg_open5))
    chosen_neg = neg_alt[:n_alt] + neg_open5[:n_open5]
    if len(chosen_neg) < half:
        extra = neg_alt[n_alt:] + neg_open5[n_open5:]
        rng.shuffle(extra)
        chosen_neg.extend(extra[: half - len(chosen_neg)])

    rows = chosen_pos + chosen_neg[: len(chosen_pos)]
    rng.shuffle(rows)
    stats = {
        "total": len(rows),
        "pos": len(chosen_pos),
        "neg": len(chosen_neg[: len(chosen_pos)]),
        "pos_sources": dict(Counter(r["source_type"] for r in chosen_pos)),
        "neg_sources": dict(Counter(r["source_type"] for r in chosen_neg[: len(chosen_pos)])),
    }
    return rows, stats


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tar-path", type=Path, default=DEFAULT_TAR)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_STAGE3_DIR)
    parser.add_argument("--max-containment", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 1. Cargar 20 genomas Test congelados y calcular sus sketches de 21-mers
    with (args.test_dir / "manifiesto_piloto.tsv").open(encoding="utf-8") as fh:
        test_manifest = list(csv.DictReader(fh, delimiter="\t"))
    test_accs = {r["accession"] for r in test_manifest}
    test_sketches: dict[str, set[bytes]] = {}
    for acc in sorted(test_accs):
        rec = next(SeqIO.parse(str(args.test_dir / "genomas" / f"{acc}.fna"), "fasta"))
        test_sketches[acc] = kmer21_sketch(str(rec.seq).upper())
    print(f"Test congelado: {len(test_sketches)} genomas cargados.")

    # 2. Indexar miembros del tar sin compresión y seleccionar Train (120) + Val (20) sin fuga de homología
    fna_members: dict[str, tarfile.TarInfo] = {}
    gff_members: dict[str, tarfile.TarInfo] = {}
    with tarfile.open(args.tar_path, "r:") as tf:
        for member in tf:
            if not member.isfile():
                continue
            parts = member.name.lstrip("./").split("/")
            if len(parts) != 3 or parts[0] != "GenomesDB":
                continue
            acc, fname = parts[1], parts[2]
            if fname == f"{acc}.fna":
                fna_members[acc] = member
            elif fname == f"{acc}.gff":
                gff_members[acc] = member

        candidate_accs = sorted((set(fna_members) & set(gff_members)) - test_accs)
        rng = random.Random(args.seed)
        rng.shuffle(candidate_accs)

        train_by_stratum: dict[str, list[dict]] = defaultdict(list)
        val_by_stratum: dict[str, list[dict]] = defaultdict(list)
        train_sketches: dict[str, set[bytes]] = {}
        val_sketches: dict[str, set[bytes]] = {}

        excluded_test_hom = 0
        excluded_train_hom = 0
        excluded_quality = 0
        scanned = 0
        max_cont_train_test = 0.0
        max_cont_val_test = 0.0
        max_cont_val_train = 0.0

        def all_filled() -> bool:
            return (
                all(len(train_by_stratum[k]) >= v for k, v in TRAIN_TARGETS.items())
                and all(len(val_by_stratum[k]) >= v for k, v in VAL_TARGETS.items())
            )

        for acc in candidate_accs:
            if all_filled():
                break
            scanned += 1
            fna_bytes = tf.extractfile(fna_members[acc]).read()
            records = parse_fasta_text(fna_bytes.decode("utf-8", errors="replace"))
            if len(records) != 1:
                excluded_quality += 1
                continue
            rec_id, seq_u = records[0]
            if set(seq_u) - ALLOWED_ACGTN:
                excluded_quality += 1
                continue
            st = size_stratum(len(seq_u))
            need_train = len(train_by_stratum[st]) < TRAIN_TARGETS[st]
            need_val = len(val_by_stratum[st]) < VAL_TARGETS[st]
            if not need_train and not need_val:
                continue

            gff_bytes = tf.extractfile(gff_members[acc]).read()
            cds_list = parse_gff_cds(gff_bytes.decode("utf-8", errors="replace"), rec_id)
            if len(cds_list) < 5:
                excluded_quality += 1
                continue

            sk = kmer21_sketch(seq_u)
            if not sk:
                excluded_quality += 1
                continue

            # Verificar contención contra los 20 genomas de Test
            c_test = max(containment(sk, t_sk) for t_sk in test_sketches.values())
            if c_test >= args.max_containment:
                excluded_test_hom += 1
                continue

            # Primero intentar asignar a Train si falta en ese estrato;
            # pero ojo: para que Val también tenga contención < 0.05 contra Train,
            # si este genoma va a Val comprobamos contra los Train ya aceptados,
            # y al final verificaremos TODOS los pares (Val, Train)!
            if need_train:
                # Si ya hay algunos Val elegidos, asegurarse también contra Val
                c_val_cur = max((containment(sk, v_sk) for v_sk in val_sketches.values()), default=0.0)
                if c_val_cur >= args.max_containment:
                    excluded_train_hom += 1
                    continue
                train_by_stratum[st].append(
                    {"accession": acc, "seq": seq_u, "cds": cds_list, "fna_bytes": fna_bytes, "gff_bytes": gff_bytes}
                )
                train_sketches[acc] = sk
                max_cont_train_test = max(max_cont_train_test, c_test)
            elif need_val:
                c_tr = max(containment(sk, tr_sk) for tr_sk in train_sketches.values())
                if c_tr >= args.max_containment:
                    excluded_train_hom += 1
                    continue
                val_by_stratum[st].append(
                    {"accession": acc, "seq": seq_u, "cds": cds_list, "fna_bytes": fna_bytes, "gff_bytes": gff_bytes}
                )
                val_sketches[acc] = sk
                max_cont_val_test = max(max_cont_val_test, c_test)
                max_cont_val_train = max(max_cont_val_train, c_tr)

    # Verificar que se cubrieron todos los estratos
    for st, req in TRAIN_TARGETS.items():
        if len(train_by_stratum[st]) < req:
            raise RuntimeError(f"Train estrato {st} incompleto: {len(train_by_stratum[st])} < {req}")
    for st, req in VAL_TARGETS.items():
        if len(val_by_stratum[st]) < req:
            raise RuntimeError(f"Val estrato {st} incompleto: {len(val_by_stratum[st])} < {req}")

    train_genomes = [g for st in TRAIN_TARGETS for g in train_by_stratum[st]]
    val_genomes = [g for st in VAL_TARGETS for g in val_by_stratum[st]]
    train_accs = {g["accession"] for g in train_genomes}
    val_accs = {g["accession"] for g in val_genomes}

    # Aserciones estrictas de 0 fuga por identidad y homología
    assert not (train_accs & val_accs)
    assert not (train_accs & test_accs)
    assert not (val_accs & test_accs)

    max_cont_val_train_exact = max(
        containment(val_sketches[va], train_sketches[ta])
        for va in val_accs
        for ta in train_accs
    )
    max_cont_val_test_exact = max(
        containment(val_sketches[va], test_sketches[tea])
        for va in val_accs
        for tea in test_accs
    )
    max_cont_train_test_exact = max(
        containment(train_sketches[ta], test_sketches[tea])
        for ta in train_accs
        for tea in test_accs
    )
    assert max_cont_train_test_exact < args.max_containment
    assert max_cont_val_test_exact < args.max_containment
    assert max_cont_val_train_exact < args.max_containment

    print(
        f"Split verificado: Train={len(train_genomes)} Val={len(val_genomes)} Test={len(test_accs)} | "
        f"scanned={scanned} excl_test_hom={excluded_test_hom} excl_train_hom={excluded_train_hom} | "
        f"max_cont(Tr,Te)={max_cont_train_test_exact:.4f} max_cont(Va,Te)={max_cont_val_test_exact:.4f} "
        f"max_cont(Va,Tr)={max_cont_val_train_exact:.4f}"
    )

    # 3. Guardar genomas extraídos de Train y Val
    for split_name, glist in (("train", train_genomes), ("val", val_genomes)):
        gdir = args.out_dir / "genomas_train_val" / split_name
        gdir.mkdir(parents=True, exist_ok=True)
        man_rows = []
        for g in glist:
            acc = g["accession"]
            (gdir / f"{acc}.fna").write_bytes(g["fna_bytes"])
            (gdir / f"{acc}.gff").write_bytes(g["gff_bytes"])
            man_rows.append(
                {
                    "accession": acc,
                    "length_bp": len(g["seq"]),
                    "stratum": size_stratum(len(g["seq"])),
                    "n_cds": len(g["cds"]),
                }
            )
        with (gdir / f"manifiesto_{split_name}.tsv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["accession", "length_bp", "stratum", "n_cds"], delimiter="\t")
            w.writeheader()
            w.writerows(man_rows)

    # 4. Generar fragmentos y pools CDS / TIS para Train y Val
    print("Generando fragmentos y ejemplos CDS/TIS para Train (120 genomas)...")
    tr_cds_pos, tr_cds_neg, tr_tis_pos, tr_tis_neg = build_cds_examples_from_genomes(
        train_genomes, TRAIN_WINDOW_CAPS, args.seed, save_fragments_dir=None, supplement_short=True
    )
    print("Generando fragmentos y ejemplos CDS/TIS para Val (20 genomas)...")
    va_cds_pos, va_cds_neg, va_tis_pos, va_tis_neg = build_cds_examples_from_genomes(
        val_genomes,
        VAL_WINDOW_CAPS,
        args.seed,
        save_fragments_dir=args.out_dir / "val_fragmentos",
        supplement_short=False,
    )

    # 5. Muestrear y guardar CSVs balanceados para FagoLM-CDS y FagoLM-TIS
    cds_train_rows, cds_train_stats = sample_balanced_cds(
        tr_cds_pos, tr_cds_neg, total_target=28_000, seed=args.seed, allow_supp_neg_both=True
    )
    cds_val_rows, cds_val_stats = sample_balanced_cds(
        va_cds_pos, va_cds_neg, total_target=5_000, seed=args.seed + 1, allow_supp_neg_both=False
    )

    cds_fields = ["sequence", "label", "category", "len_bucket", "record_id"]
    cds_dir = args.out_dir / "dataset_cds"
    write_csv(cds_dir / "train.csv", cds_train_rows, cds_fields)
    write_csv(cds_dir / "val.csv", cds_val_rows, cds_fields)
    (cds_dir / "data_index.json").write_text(
        json.dumps({"train": ["train.csv"], "dev": ["val.csv"], "test": []}, indent=2) + "\n",
        encoding="utf-8",
    )

    tis_train_rows, tis_train_stats = sample_balanced_tis(
        tr_tis_pos, tr_tis_neg, total_target=28_000, seed=args.seed
    )
    tis_val_rows, tis_val_stats = sample_balanced_tis(
        va_tis_pos, va_tis_neg, total_target=5_000, seed=args.seed + 1
    )

    # Verificar que el 100% de las ventanas TIS tienen exactamente 60 pb
    assert all(len(r["sequence"]) == 60 for r in tis_train_rows)
    assert all(len(r["sequence"]) == 60 for r in tis_val_rows)

    tis_fields = ["sequence", "label", "source_type", "record_id"]
    tis_dir = args.out_dir / "dataset_tis"
    write_csv(tis_dir / "train.csv", tis_train_rows, tis_fields)
    write_csv(tis_dir / "val.csv", tis_val_rows, tis_fields)
    (tis_dir / "data_index.json").write_text(
        json.dumps({"train": ["train.csv"], "dev": ["val.csv"], "test": []}, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {
        "seed": args.seed,
        "max_containment_threshold": args.max_containment,
        "tar_scanned_candidates": scanned,
        "excluded_test_homology": excluded_test_hom,
        "excluded_train_homology": excluded_train_hom,
        "excluded_quality": excluded_quality,
        "max_containment_train_test": round(max_cont_train_test_exact, 6),
        "max_containment_val_test": round(max_cont_val_test_exact, 6),
        "max_containment_val_train": round(max_cont_val_train_exact, 6),
        "train_genomes": len(train_genomes),
        "val_genomes": len(val_genomes),
        "test_genomes": len(test_accs),
        "cds_raw_pools": {
            "train_pos": len(tr_cds_pos),
            "train_neg": len(tr_cds_neg),
            "val_pos": len(va_cds_pos),
            "val_neg": len(va_cds_neg),
        },
        "tis_raw_pools": {
            "train_pos": len(tr_tis_pos),
            "train_neg": len(tr_tis_neg),
            "val_pos": len(va_tis_pos),
            "val_neg": len(va_tis_neg),
        },
        "dataset_cds_train": cds_train_stats,
        "dataset_cds_val": cds_val_stats,
        "dataset_tis_train": tis_train_stats,
        "dataset_tis_val": tis_val_stats,
    }
    (args.out_dir / "resumen_split_etapa3.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("Resumen CDS Train:", json.dumps(cds_train_stats))
    print("Resumen CDS Val:", json.dumps(cds_val_stats))
    print("Resumen TIS Train:", json.dumps(tis_train_stats))
    print("Resumen TIS Val:", json.dumps(tis_val_stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
