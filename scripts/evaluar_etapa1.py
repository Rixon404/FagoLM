#!/usr/bin/env python3
"""Evalúa cuantitativamente las predicciones de los 3 modos de GeneLM (Etapa 1)
contra las referencias GFF de INPHARED, incluyendo embudo por etapa y desgloses
por hebra, longitud de CDS, soporte PHROG y estrato genómico.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

_REPO_DIR = Path(__file__).resolve().parent.parent
_PARENT_DIR = _REPO_DIR.parent

from Bio import SeqIO
from Bio.Seq import Seq

DEFAULT_PILOTO = (_REPO_DIR / "etapa1_fagos_completos/piloto_inphared"
)
DEFAULT_PRED = (_REPO_DIR / "etapa1_fagos_completos/predicciones"
)
DEFAULT_OUT = (_REPO_DIR / "etapa1_fagos_completos/resultados"
)

MODES = (
    "modo_a_original",
    "modo_b_adaptado_fagos",
    "modo_c_diagnostico_hebra_menos",
)


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def calc_f1(p: float, r: float) -> float:
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def parse_gff_records(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            if line.strip() == "##FASTA":
                break
            continue
        parts = line.split("\t")
        if len(parts) != 9:
            continue
        seqid, source, ftype, s1_s, e1_s, score, strand, phase, attrs = parts
        if ftype.lower() not in {"cds", "orf"} or strand not in {"+", "-"}:
            continue
        s1, e1 = int(s1_s), int(e1_s)
        s0, e0 = s1 - 1, e1
        stop_coord_0 = e0 if strand == "+" else s0
        rows.append(
            {
                "seqid": seqid,
                "start_0": s0,
                "end_0": e0,
                "start_1": s1,
                "end_1": e1,
                "length_bp": e0 - s0,
                "strand": strand,
                "stop_coord_0": stop_coord_0,
                "has_phrog": ("phrog_" in attrs.lower()),
            }
        )
    return rows


def read_tsv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


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
            if r["strand"] != p["strand"]:
                continue
            iou = interval_iou(r["start_0"], r["end_0"], p["start_0"], p["end_0"])
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
    ref_to_pred = {ri: pi for pi, ri in pred_to_ref.items()}
    return ref_to_pred, ious


def match_by_stop(refs: list[dict], preds: list[dict]):
    """Empareja referencias y predicciones que comparten exactamente (strand, stop_coord_0)."""
    refs_by_key: dict[tuple[str, int], list[int]] = defaultdict(list)
    preds_by_key: dict[tuple[str, int], list[int]] = defaultdict(list)
    for ri, r in enumerate(refs):
        refs_by_key[(r["strand"], r["stop_coord_0"])].append(ri)
    for pi, p in enumerate(preds):
        preds_by_key[(p["strand"], p["stop_coord_0"])].append(pi)

    ref_to_pred: dict[int, int] = {}
    for key, r_idxs in refs_by_key.items():
        p_idxs = preds_by_key.get(key, [])
        if not p_idxs:
            continue
        used_p: set[int] = set()
        # Primero coincidencias exactas de inicio+fin
        for ri in r_idxs:
            r = refs[ri]
            for pi in p_idxs:
                if pi in used_p:
                    continue
                p = preds[pi]
                if p["start_0"] == r["start_0"] and p["end_0"] == r["end_0"]:
                    ref_to_pred[ri] = pi
                    used_p.add(pi)
                    break
        # Luego emparejar restantes por stop
        for ri in r_idxs:
            if ri in ref_to_pred:
                continue
            for pi in p_idxs:
                if pi not in used_p:
                    ref_to_pred[ri] = pi
                    used_p.add(pi)
                    break
    return ref_to_pred


def length_bin(length_bp: int) -> str:
    if length_bp <= 100:
        return "<=100"
    if length_bp <= 300:
        return "101-300"
    if length_bp <= 900:
        return "301-900"
    return ">900"


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--piloto-dir", type=Path, default=DEFAULT_PILOTO)
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-iou", type=float, default=0.5)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_tsv(args.piloto_dir / "manifiesto_piloto.tsv")

    summary_modes_rows: list[dict] = []
    genome_rows: list[dict] = []
    strand_rows: list[dict] = []
    length_rows: list[dict] = []
    phrog_rows: list[dict] = []
    stratum_rows: list[dict] = []
    funnel_rows: list[dict] = []

    for mode in MODES:
        mode_dir = args.pred_dir / mode
        runtime_path = mode_dir / "runtime_stats.json"
        runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.is_file() else {}

        tot = Counter()
        strand_acc = defaultdict(Counter)
        len_acc = defaultdict(Counter)
        phrog_acc = defaultdict(Counter)
        stratum_acc = defaultdict(Counter)
        funnel_acc = Counter()

        for mrow in manifest:
            acc = mrow["accession"]
            stratum = mrow["stratum"]
            genome_bp = int(mrow["length_bp"])
            refs = parse_gff_records(args.piloto_dir / "genomas" / f"{acc}.gff")
            preds = parse_gff_records(mode_dir / f"{acc}.gff")
            cds_groups = read_tsv(mode_dir / f"{acc}_etapa1_cds_groups.tsv")
            tis_cands = read_tsv(mode_dir / f"{acc}_etapa2_tis_candidates.tsv")

            # Verificar secuencia real de cada referencia para saber si tiene start/stop permitido por el modo
            rec = next(SeqIO.parse(str(args.piloto_dir / "genomas" / f"{acc}.fna"), "fasta"))
            seq_u = str(rec.seq).upper()

            # Mapas de Etapa 1 y Etapa 2
            # En etapa1_cds_groups.tsv: stop_coord_1 es end_1 en + y start_1 en -
            # Normalicemos a stop_coord_0 = int(stop_coord_1) if '+' else int(stop_coord_1) - 1
            stage1_by_stop: dict[tuple[str, int], dict] = {}
            for g in cds_groups:
                st = g["strand"]
                sc0 = int(g["stop_coord_1"]) if st == "+" else int(g["stop_coord_1"]) - 1
                stage1_by_stop[(st, sc0)] = g

            tis_by_exact: dict[tuple[str, int, int], dict] = {}
            for c in tis_cands:
                tis_by_exact[(c["strand"], int(c["start_1"]) - 1, int(c["end_1"]))] = c

            ref_to_pred_iou, _ = maximum_matching_iou(refs, preds, min_iou=args.min_iou)
            ref_to_pred_stop = match_by_stop(refs, preds)

            # Métricas por genoma
            n_refs = len(refs)
            n_preds = len(preds)
            tp_iou = len(ref_to_pred_iou)
            fp_iou = n_preds - tp_iou
            fn_iou = n_refs - tp_iou
            exact_iou = sum(
                1
                for ri, pi in ref_to_pred_iou.items()
                if refs[ri]["start_0"] == preds[pi]["start_0"]
                and refs[ri]["end_0"] == preds[pi]["end_0"]
            )

            stop_tp = len(ref_to_pred_stop)
            exact_tp = sum(
                1
                for ri, pi in ref_to_pred_stop.items()
                if refs[ri]["start_0"] == preds[pi]["start_0"]
                and refs[ri]["end_0"] == preds[pi]["end_0"]
            )

            # Verificar consistencia aritmética por genoma
            assert tp_iou + fn_iou == n_refs
            assert tp_iou + fp_iou == n_preds
            assert stop_tp >= exact_tp

            p_iou = safe_div(tp_iou, n_preds)
            r_iou = safe_div(tp_iou, n_refs)
            f1_iou = calc_f1(p_iou, r_iou)
            stop_p = safe_div(stop_tp, n_preds)
            stop_r = safe_div(stop_tp, n_refs)
            stop_f1 = calc_f1(stop_p, stop_r)

            genome_rows.append(
                {
                    "mode": mode,
                    "accession": acc,
                    "stratum": stratum,
                    "length_bp": genome_bp,
                    "refs": n_refs,
                    "preds": n_preds,
                    "tp_iou": tp_iou,
                    "fp_iou": fp_iou,
                    "fn_iou": fn_iou,
                    "precision_iou": round(p_iou, 4),
                    "recall_iou": round(r_iou, 4),
                    "f1_iou": round(f1_iou, 4),
                    "stop_tp": stop_tp,
                    "stop_precision": round(stop_p, 4),
                    "stop_recall": round(stop_r, 4),
                    "stop_f1": round(stop_f1, 4),
                    "exact_tp": exact_tp,
                    "exact_recall": round(safe_div(exact_tp, n_refs), 4),
                    "tis_acc_cond": round(safe_div(exact_tp, stop_tp), 4),
                    "fp_per_kb": round(safe_div(fp_iou, genome_bp / 1000.0), 4),
                }
            )

            tot["bp"] += genome_bp
            tot["refs"] += n_refs
            tot["preds"] += n_preds
            tot["tp_iou"] += tp_iou
            tot["fp_iou"] += fp_iou
            tot["fn_iou"] += fn_iou
            tot["exact_iou"] += exact_iou
            tot["stop_tp"] += stop_tp
            tot["exact_tp"] += exact_tp

            # Conteo de precisión de Etapa 1 CDS (grupos stop pred==1 que existen en refs)
            ref_stop_keys = {(r["strand"], r["stop_coord_0"]) for r in refs}
            for g in cds_groups:
                st = g["strand"]
                sc0 = int(g["stop_coord_1"]) if st == "+" else int(g["stop_coord_1"]) - 1
                if int(g["cds_pred"]) == 1:
                    funnel_acc["stage1_total_pos_groups"] += 1
                    if (st, sc0) in ref_stop_keys:
                        funnel_acc["stage1_tp_groups"] += 1

            # Conteo por predicción para precisiones por hebra y estrato
            matched_pred_stop = set(ref_to_pred_stop.values())
            for pi, p in enumerate(preds):
                strand_acc[p["strand"]]["preds"] += 1
                stratum_acc[stratum]["preds"] += 1
                if pi in matched_pred_stop:
                    strand_acc[p["strand"]]["pred_stop_tp"] += 1
                    stratum_acc[stratum]["pred_stop_tp"] += 1

            matched_pred_iou = set(ref_to_pred_iou.values())
            for pi, p in enumerate(preds):
                if pi in matched_pred_iou:
                    stratum_acc[stratum]["pred_iou_tp"] += 1

            stratum_acc[stratum]["bp"] += genome_bp

            # Acumuladores por referencia (embudo + desgloses)
            for ri, r in enumerate(refs):
                st = r["strand"]
                lbin = length_bin(r["length_bp"])
                pbin = "with_phrog" if r["has_phrog"] else "without_phrog"
                skey = (st, r["stop_coord_0"])
                ekey = (st, r["start_0"], r["end_0"])

                is_iou = int(ri in ref_to_pred_iou)
                is_stop = int(ri in ref_to_pred_stop)
                is_exact = int(
                    ri in ref_to_pred_stop
                    and preds[ref_to_pred_stop[ri]]["start_0"] == r["start_0"]
                    and preds[ref_to_pred_stop[ri]]["end_0"] == r["end_0"]
                )

                g_entry = stage1_by_stop.get(skey)
                in_extractor_stop = int(g_entry is not None)
                stage1_pos = int(g_entry is not None and int(g_entry["cds_pred"]) == 1)

                t_entry = tis_by_exact.get(ekey)
                tis_evaluated = int(t_entry is not None)
                tis_empty_win = int(t_entry is not None and int(t_entry["tis_window_len"]) == 0)
                tis_trunc_win = int(t_entry is not None and 0 < int(t_entry["tis_window_len"]) < 60)
                tis_pred1 = int(t_entry is not None and int(t_entry["tis_pred_classic"]) == 1)

                funnel_acc["refs"] += 1
                funnel_acc["extractor_stop"] += in_extractor_stop
                funnel_acc["stage1_cds_pos"] += stage1_pos
                funnel_acc["stage2_tis_evaluated"] += tis_evaluated
                funnel_acc["stage2_empty_win"] += tis_empty_win
                funnel_acc["stage2_trunc_win"] += tis_trunc_win
                funnel_acc["stage2_tis_pred1"] += tis_pred1
                funnel_acc["final_stop_tp"] += is_stop
                funnel_acc["final_exact_tp"] += is_exact

                for acc_dict, key in (
                    (strand_acc, st),
                    (len_acc, lbin),
                    (phrog_acc, pbin),
                    (stratum_acc, stratum),
                ):
                    acc_dict[key]["refs"] += 1
                    acc_dict[key]["iou_tp"] += is_iou
                    acc_dict[key]["stop_tp"] += is_stop
                    acc_dict[key]["exact_tp"] += is_exact
                    acc_dict[key]["stage1_pos"] += stage1_pos
                    acc_dict[key]["extractor_stop"] += in_extractor_stop

        # Totales del modo
        assert tot["tp_iou"] + tot["fn_iou"] == tot["refs"]
        assert tot["tp_iou"] + tot["fp_iou"] == tot["preds"]
        assert tot["stop_tp"] >= tot["exact_tp"]

        p_iou = safe_div(tot["tp_iou"], tot["preds"])
        r_iou = safe_div(tot["tp_iou"], tot["refs"])
        f1_iou = calc_f1(p_iou, r_iou)
        stop_p = safe_div(tot["stop_tp"], tot["preds"])
        stop_r = safe_div(tot["stop_tp"], tot["refs"])
        stop_f1 = calc_f1(stop_p, stop_r)
        exact_p = safe_div(tot["exact_tp"], tot["preds"])
        exact_r = safe_div(tot["exact_tp"], tot["refs"])
        exact_f1 = calc_f1(exact_p, exact_r)

        summary_modes_rows.append(
            {
                "mode": mode,
                "genomes": len(manifest),
                "total_bp": tot["bp"],
                "refs": tot["refs"],
                "preds": tot["preds"],
                "tp_iou": tot["tp_iou"],
                "fp_iou": tot["fp_iou"],
                "fn_iou": tot["fn_iou"],
                "precision_iou": round(p_iou, 4),
                "recall_iou": round(r_iou, 4),
                "f1_iou": round(f1_iou, 4),
                "stop_tp": tot["stop_tp"],
                "stop_precision": round(stop_p, 4),
                "stop_recall": round(stop_r, 4),
                "stop_f1": round(stop_f1, 4),
                "exact_tp": tot["exact_tp"],
                "exact_precision": round(exact_p, 4),
                "exact_recall": round(exact_r, 4),
                "exact_f1": round(exact_f1, 4),
                "tis_acc_cond": round(safe_div(tot["exact_tp"], tot["stop_tp"]), 4),
                "fp_per_kb": round(safe_div(tot["fp_iou"], tot["bp"] / 1000.0), 4),
                "stage1_cds_recall": round(safe_div(funnel_acc["stage1_cds_pos"], tot["refs"]), 4),
                "stage1_cds_precision": round(
                    safe_div(funnel_acc["stage1_tp_groups"], funnel_acc["stage1_total_pos_groups"]), 4
                ),
                "elapsed_seconds": runtime.get("elapsed_seconds", ""),
                "peak_vram_mib": runtime.get("peak_vram_mib", ""),
            }
        )

        funnel_rows.append(
            {
                "mode": mode,
                "refs": funnel_acc["refs"],
                "extractor_stop": funnel_acc["extractor_stop"],
                "pct_extractor_stop": round(100 * safe_div(funnel_acc["extractor_stop"], funnel_acc["refs"]), 2),
                "stage1_cds_pos": funnel_acc["stage1_cds_pos"],
                "pct_stage1_cds_pos": round(100 * safe_div(funnel_acc["stage1_cds_pos"], funnel_acc["refs"]), 2),
                "stage2_tis_evaluated": funnel_acc["stage2_tis_evaluated"],
                "stage2_empty_win": funnel_acc["stage2_empty_win"],
                "stage2_trunc_win": funnel_acc["stage2_trunc_win"],
                "stage2_tis_pred1": funnel_acc["stage2_tis_pred1"],
                "pct_stage2_tis_pred1": round(100 * safe_div(funnel_acc["stage2_tis_pred1"], funnel_acc["refs"]), 2),
                "final_stop_tp": funnel_acc["final_stop_tp"],
                "pct_final_stop_tp": round(100 * safe_div(funnel_acc["final_stop_tp"], funnel_acc["refs"]), 2),
                "final_exact_tp": funnel_acc["final_exact_tp"],
                "pct_final_exact_tp": round(100 * safe_div(funnel_acc["final_exact_tp"], funnel_acc["refs"]), 2),
                "stage1_total_pos_groups": funnel_acc["stage1_total_pos_groups"],
                "stage1_tp_groups": funnel_acc["stage1_tp_groups"],
                "stage1_precision": round(
                    safe_div(funnel_acc["stage1_tp_groups"], funnel_acc["stage1_total_pos_groups"]), 4
                ),
            }
        )

        for st in ("+", "-"):
            d = strand_acc[st]
            sp = safe_div(d["pred_stop_tp"], d["preds"])
            sr = safe_div(d["stop_tp"], d["refs"])
            strand_rows.append(
                {
                    "mode": mode,
                    "strand": st,
                    "refs": d["refs"],
                    "preds": d["preds"],
                    "stage1_pos": d["stage1_pos"],
                    "stage1_recall": round(safe_div(d["stage1_pos"], d["refs"]), 4),
                    "stop_tp": d["stop_tp"],
                    "stop_precision": round(sp, 4),
                    "stop_recall": round(sr, 4),
                    "stop_f1": round(calc_f1(sp, sr), 4),
                    "exact_tp": d["exact_tp"],
                    "exact_recall": round(safe_div(d["exact_tp"], d["refs"]), 4),
                    "tis_acc_cond": round(safe_div(d["exact_tp"], d["stop_tp"]), 4),
                }
            )

        for lbin in ("<=100", "101-300", "301-900", ">900"):
            d = len_acc[lbin]
            length_rows.append(
                {
                    "mode": mode,
                    "length_bin_nt": lbin,
                    "refs": d["refs"],
                    "extractor_stop": d["extractor_stop"],
                    "stage1_pos": d["stage1_pos"],
                    "stage1_recall": round(safe_div(d["stage1_pos"], d["refs"]), 4),
                    "stop_tp": d["stop_tp"],
                    "stop_recall": round(safe_div(d["stop_tp"], d["refs"]), 4),
                    "exact_tp": d["exact_tp"],
                    "exact_recall": round(safe_div(d["exact_tp"], d["refs"]), 4),
                    "tis_acc_cond": round(safe_div(d["exact_tp"], d["stop_tp"]), 4),
                }
            )

        for pbin in ("with_phrog", "without_phrog"):
            d = phrog_acc[pbin]
            phrog_rows.append(
                {
                    "mode": mode,
                    "phrog_status": pbin,
                    "refs": d["refs"],
                    "stage1_pos": d["stage1_pos"],
                    "stage1_recall": round(safe_div(d["stage1_pos"], d["refs"]), 4),
                    "stop_tp": d["stop_tp"],
                    "stop_recall": round(safe_div(d["stop_tp"], d["refs"]), 4),
                    "exact_tp": d["exact_tp"],
                    "exact_recall": round(safe_div(d["exact_tp"], d["refs"]), 4),
                    "tis_acc_cond": round(safe_div(d["exact_tp"], d["stop_tp"]), 4),
                }
            )

        for st_name in ("lt_20kb", "20kb_50kb", "50kb_100kb", "ge_100kb"):
            d = stratum_acc[st_name]
            p_i = safe_div(d["pred_iou_tp"], d["preds"])
            r_i = safe_div(d["iou_tp"], d["refs"])
            sp = safe_div(d["pred_stop_tp"], d["preds"])
            sr = safe_div(d["stop_tp"], d["refs"])
            stratum_rows.append(
                {
                    "mode": mode,
                    "stratum": st_name,
                    "refs": d["refs"],
                    "preds": d["preds"],
                    "precision_iou": round(p_i, 4),
                    "recall_iou": round(r_i, 4),
                    "f1_iou": round(calc_f1(p_i, r_i), 4),
                    "stop_precision": round(sp, 4),
                    "stop_recall": round(sr, 4),
                    "stop_f1": round(calc_f1(sp, sr), 4),
                    "exact_recall": round(safe_div(d["exact_tp"], d["refs"]), 4),
                    "tis_acc_cond": round(safe_div(d["exact_tp"], d["stop_tp"]), 4),
                }
            )

    write_tsv(
        args.out_dir / "resumen_modos.tsv",
        summary_modes_rows,
        list(summary_modes_rows[0].keys()),
    )
    write_tsv(
        args.out_dir / "resumen_por_genoma.tsv",
        genome_rows,
        list(genome_rows[0].keys()),
    )
    write_tsv(
        args.out_dir / "desglose_hebra.tsv",
        strand_rows,
        list(strand_rows[0].keys()),
    )
    write_tsv(
        args.out_dir / "desglose_longitud.tsv",
        length_rows,
        list(length_rows[0].keys()),
    )
    write_tsv(
        args.out_dir / "desglose_phrog.tsv",
        phrog_rows,
        list(phrog_rows[0].keys()),
    )
    write_tsv(
        args.out_dir / "desglose_estrato.tsv",
        stratum_rows,
        list(stratum_rows[0].keys()),
    )
    write_tsv(
        args.out_dir / "embudo_etapas.tsv",
        funnel_rows,
        list(funnel_rows[0].keys()),
    )

    # Generar reporte Markdown
    def to_md_table(rows: list[dict], cols: list[str]) -> str:
        hdr = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        body = ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
        return "\n".join([hdr, sep, *body])

    md = [
        "# Reporte Experimental — Etapa 1: Evaluación de GeneLM en Genomas Completos de Fagos (Piloto INPHARED)",
        "",
        "## 1. Configuración del Experimento Piloto",
        f"- **Corpus origen:** `GenomesDB_Jan_2026.tar.gz (INPHARED)` (`2,977` genomas mono-contig con `.fna` + `.gff` generado por `Prodigal:002006` + PHROGs).",
        f"- **Muestra piloto:** `20` genomas estratificados por longitud (`<20 kb`: 4, `20–50 kb`: 6, `50–100 kb`: 6, `>=100 kb`: 4; semilla `42`), sumando `{summary_modes_rows[0]['total_bp']:,}` pb y `{summary_modes_rows[0]['refs']:,}` CDS de referencia.",
        "- **Checkpoints evaluados (sin *fine-tuning*):** `datasets/genelm_cds` (`BacteriaCDS-DNABERT-K6-89M`) y `datasets/genelm_tis` (`BacteriaTIS-DNABERT-K6-89M`), inferencia en GPU (`RTX 3050 Ti 4 GB`, FP16 verificado con 100% de concordancia frente a FP32 en el primer genoma).",
        "- **Modos comparados:**",
        "  - `modo_a_original`: Replica las reglas de `core.py` (`starts=['TTG','CTG','ATG','GTG']`, `length_coding > 100`, `seq[-510:]` en ambas hebras, recorte TIS sin guarda en bordes).",
        "  - `modo_b_adaptado_fagos`: Adapta solo el wrapper (`starts=['ATG','GTG','TTG','ATT','ATC','ATA']`, `total_len >= 60`, padding `'N'` en ventanas TIS de bordes `<30 pb`).",
        "  - `modo_c_diagnostico_hebra_menos`: Igual a `modo_a_original`, pero recorta `seq[:510]` en hebra `-` (como `01-Analyse-CDS.ipynb` celda 16 al entrenar `BacteriaCDS`).",
        "",
        "## 2. Resultados Globales por Modo",
        to_md_table(
            summary_modes_rows,
            [
                "mode",
                "refs",
                "preds",
                "precision_iou",
                "recall_iou",
                "f1_iou",
                "stop_precision",
                "stop_recall",
                "stop_f1",
                "exact_recall",
                "tis_acc_cond",
                "fp_per_kb",
                "elapsed_seconds",
                "peak_vram_mib",
            ],
        ),
        "",
        "## 3. Embudo de Pérdidas por Etapa (Etapa 1 CDS vs Etapa 2 TIS)",
        to_md_table(
            funnel_rows,
            [
                "mode",
                "refs",
                "extractor_stop",
                "pct_extractor_stop",
                "stage1_cds_pos",
                "pct_stage1_cds_pos",
                "stage2_tis_pred1",
                "pct_stage2_tis_pred1",
                "final_stop_tp",
                "pct_final_stop_tp",
                "final_exact_tp",
                "pct_final_exact_tp",
                "stage1_precision",
                "stage2_empty_win",
                "stage2_trunc_win",
            ],
        ),
        "",
        "## 4. Desglose por Hebra (`+` vs `-`)",
        to_md_table(
            strand_rows,
            [
                "mode",
                "strand",
                "refs",
                "preds",
                "stage1_recall",
                "stop_precision",
                "stop_recall",
                "stop_f1",
                "exact_recall",
                "tis_acc_cond",
            ],
        ),
        "",
        "## 5. Desglose por Longitud de CDS (nt)",
        to_md_table(
            length_rows,
            [
                "mode",
                "length_bin_nt",
                "refs",
                "extractor_stop",
                "stage1_recall",
                "stop_recall",
                "exact_recall",
                "tis_acc_cond",
            ],
        ),
        "",
        "## 6. Desglose por Soporte Funcional PHROG",
        to_md_table(
            phrog_rows,
            [
                "mode",
                "phrog_status",
                "refs",
                "stage1_recall",
                "stop_recall",
                "exact_recall",
                "tis_acc_cond",
            ],
        ),
        "",
        "## 7. Desglose por Estrato de Tamaño Genómico",
        to_md_table(
            stratum_rows,
            [
                "mode",
                "stratum",
                "refs",
                "preds",
                "precision_iou",
                "recall_iou",
                "f1_iou",
                "stop_f1",
                "exact_recall",
                "tis_acc_cond",
            ],
        ),
    ]
    report_path = args.out_dir / "REPORTE_ETAPA1_GENELM_FAGOS.md"
    report_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print("Resumen global por modo:")
    for r in summary_modes_rows:
        print(
            f"  {r['mode']}: P_iou={r['precision_iou']} R_iou={r['recall_iou']} F1_iou={r['f1_iou']} | "
            f"Stop_P={r['stop_precision']} Stop_R={r['stop_recall']} Stop_F1={r['stop_f1']} | "
            f"Exact_R={r['exact_recall']} TIS_cond={r['tis_acc_cond']} | Stage1_R={r['stage1_cds_recall']}"
        )
    print(f"Reporte escrito en: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
