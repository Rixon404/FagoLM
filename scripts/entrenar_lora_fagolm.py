#!/usr/bin/env python3
"""Paso 3 (Etapa 3): Fine-tuning con LoRA (peft) sobre los checkpoints locales de GeneLM
para obtener FagoLM-CDS y FagoLM-TIS usando padding dinámico en GPU de 4 GB.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import time
from pathlib import Path

_REPO_DIR = Path(__file__).resolve().parent.parent
_PARENT_DIR = _REPO_DIR.parent

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

DEFAULT_STAGE3 = (_REPO_DIR / "etapa3_finetuning_lora"
)
DEFAULT_MODELS = (_PARENT_DIR / "trabajos_referencia/datasets")


def generate_kmer(sequence: str, k: int = 6, overlap: int = 1) -> str:
    seq = str(sequence).upper()
    if len(seq) < k:
        return ""
    return " ".join(seq[j : j + k] for j in range(0, len(seq) - k + 1, overlap))


def softmax_2d(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)


def compute_cls_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    p1 = probs[:, 1]
    y_pred = (probs[:, 1] >= probs[:, 0]).astype(int)
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    fp = int(np.sum((y_pred == 1) & neg_mask))
    tn = int(np.sum((y_pred == 0) & neg_mask))
    return {
        "n": int(len(y_true)),
        "n_pos": int(np.sum(pos_mask)),
        "n_neg": int(np.sum(neg_mask)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "fpr": round(float(fp / (fp + tn)) if (fp + tn) else 0.0, 4),
        "auroc": round(float(roc_auc_score(y_true, p1)), 4) if len(set(y_true.tolist())) > 1 else 0.0,
    }


def make_hf_compute_metrics():
    def compute_metrics(eval_pred):
        preds = eval_pred.predictions
        logits = preds[0] if isinstance(preds, tuple) else preds
        labels = eval_pred.label_ids
        probs = softmax_2d(logits)
        m = compute_cls_metrics(labels, probs)
        return {
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "fpr": m["fpr"],
            "auroc": m["auroc"],
        }

    return compute_metrics


def prepare_tokenized_dataset(df: pd.DataFrame, tokenizer: AutoTokenizer, overlap: int) -> Dataset:
    kmer_texts = [generate_kmer(s, k=6, overlap=overlap) for s in df["sequence"].tolist()]
    labels = [int(x) for x in df["label"].tolist()]
    ds = Dataset.from_dict({"text": kmer_texts, "labels": labels})

    def tok_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=512)

    ds = ds.map(tok_fn, batched=True, batch_size=2000, remove_columns=["text"])
    return ds


def train_one_task(task: str, args) -> dict:
    assert task in ("cds", "tis")
    set_seed(args.seed)

    base_model_dir = args.models_dir / ("genelm_cds" if task == "cds" else "genelm_tis")
    data_dir = args.stage3_dir / ("dataset_cds" if task == "cds" else "dataset_tis")
    out_dir = args.stage3_dir / "modelos" / (f"fagolm_{task}_lora")
    tmp_ckpt_dir = out_dir / "_tmp_ckpt"
    if tmp_ckpt_dir.exists():
        shutil.rmtree(tmp_ckpt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overlap = 3 if task == "cds" else 1
    df_train = pd.read_csv(data_dir / "train.csv")
    df_val = pd.read_csv(data_dir / "val.csv")

    tokenizer = AutoTokenizer.from_pretrained(str(base_model_dir), local_files_only=True)
    ds_train = prepare_tokenized_dataset(df_train, tokenizer, overlap=overlap)
    ds_val = prepare_tokenized_dataset(df_val, tokenizer, overlap=overlap)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def build_model(grad_ckpt: bool = False):
        base = AutoModelForSequenceClassification.from_pretrained(
            str(base_model_dir), local_files_only=True
        )
        cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["query", "key", "value", "dense"],
            bias="none",
            task_type=TaskType.SEQ_CLS,
        )
        m = get_peft_model(base, cfg)
        if grad_ckpt:
            m.enable_input_require_grads()
        return m

    bs_train = 16 if task == "cds" else 32
    grad_acc = 2 if task == "cds" else 1
    bs_eval = 64 if task == "cds" else 128

    t0 = time.perf_counter()
    used_fallback_bs = False

    for attempt in range(2):
        grad_ckpt = attempt == 1
        cur_bs = bs_train if attempt == 0 else 8
        cur_ga = grad_acc if attempt == 0 else 4
        if attempt == 1:
            used_fallback_bs = True
            if tmp_ckpt_dir.exists():
                shutil.rmtree(tmp_ckpt_dir)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        model = build_model(grad_ckpt=grad_ckpt)
        if attempt == 0:
            model.print_trainable_parameters()

        t_args = TrainingArguments(
            output_dir=str(tmp_ckpt_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=cur_bs,
            gradient_accumulation_steps=cur_ga,
            per_device_eval_batch_size=bs_eval,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            weight_decay=0.01,
            fp16=(device.type == "cuda"),
            gradient_checkpointing=grad_ckpt,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to="none",
            logging_steps=100,
            seed=args.seed,
            dataloader_num_workers=2,
        )
        trainer = Trainer(
            model=model,
            args=t_args,
            train_dataset=ds_train,
            eval_dataset=ds_val,
            processing_class=tokenizer,
            data_collator=collator,
            compute_metrics=make_hf_compute_metrics(),
        )
        try:
            train_out = trainer.train()
            break
        except torch.cuda.OutOfMemoryError:
            if attempt == 1:
                raise
            print(f"[{task}] CUDA OOM con batch={cur_bs}; reintentando con batch=8, grad_accum=4, gradient_checkpointing=True")
            del trainer, model

    elapsed = time.perf_counter() - t0
    peak_mib = (
        round(torch.cuda.max_memory_allocated(device) / (1024**2), 2)
        if device.type == "cuda"
        else 0.0
    )

    # Guardar mejor adaptador PEFT + tokenizador y limpiar checkpoints intermedios
    trainer.model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # Evaluar desglose detallado sobre val.csv usando el mejor modelo cargado
    val_pred_out = trainer.predict(ds_val)
    val_logits = (
        val_pred_out.predictions[0]
        if isinstance(val_pred_out.predictions, tuple)
        else val_pred_out.predictions
    )
    val_probs = softmax_2d(val_logits)
    val_labels = np.array(df_val["label"].tolist(), dtype=int)
    val_overall = compute_cls_metrics(val_labels, val_probs)

    val_breakdown: dict[str, dict] = {}
    if task == "cds":
        seq_lens = np.array([len(str(s)) for s in df_val["sequence"].tolist()])
        for name, mask in (("len_le_103", seq_lens <= 103), ("len_gt_103", seq_lens > 103)):
            if np.any(mask):
                val_breakdown[name] = compute_cls_metrics(val_labels[mask], val_probs[mask])
        if "region_kind" in df_val.columns:
            rk_series = df_val["region_kind"]
        else:
            rk_series = df_val["category"].astype(str).map(lambda c: c.replace("neg_", "", 1).split("__")[0])
        for rk in sorted(rk_series.unique()):
            mask = (rk_series == rk).to_numpy()
            if np.any(mask):
                val_breakdown[f"region_{rk}"] = compute_cls_metrics(val_labels[mask], val_probs[mask])
    else:
        for st_val in sorted(df_val["source_type"].unique()):
            mask = (df_val["source_type"] == st_val).to_numpy()
            if np.any(mask):
                y_sub = val_labels[mask]
                p_sub = val_probs[mask]
                y_hat = (p_sub[:, 1] >= p_sub[:, 0]).astype(int)
                val_breakdown[st_val] = {
                    "n": int(np.sum(mask)),
                    "acc": round(float(np.mean(y_hat == y_sub)), 4),
                    "mean_prob1": round(float(np.mean(p_sub[:, 1])), 4),
                }

    history = [
        {k: v for k, v in log.items() if k in ("epoch", "loss", "eval_loss", "eval_accuracy", "eval_precision", "eval_recall", "eval_f1", "eval_fpr", "eval_auroc")}
        for log in trainer.state.log_history
    ]

    del trainer, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if tmp_ckpt_dir.exists():
        shutil.rmtree(tmp_ckpt_dir)

    # Verificar recarga limpia del adaptador LoRA + merge_and_unload
    reloaded_base = AutoModelForSequenceClassification.from_pretrained(
        str(base_model_dir), local_files_only=True
    )
    merged = PeftModel.from_pretrained(reloaded_base, str(out_dir)).merge_and_unload()
    reload_ok = isinstance(merged, torch.nn.Module)
    del merged, reloaded_base
    gc.collect()

    metrics_payload = {
        "task": task,
        "base_model": str(base_model_dir),
        "adapter_dir": str(out_dir),
        "train_rows": len(df_train),
        "val_rows": len(df_val),
        "epochs": args.epochs,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "used_oom_fallback": used_fallback_bs,
        "train_loss_final": round(float(train_out.training_loss), 4),
        "elapsed_seconds": round(elapsed, 2),
        "peak_vram_mib": peak_mib,
        "val_overall": val_overall,
        "val_breakdown": val_breakdown,
        "reload_merge_ok": reload_ok,
        "log_history": history,
    }
    (out_dir / "training_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[{task.upper()}] done in {elapsed:.1f}s | peak_vram={peak_mib} MiB | "
        f"val={val_overall} | reload_ok={reload_ok}"
    )
    if task == "cds":
        print(f"[{task.upper()}] val_breakdown:", json.dumps(val_breakdown))
    else:
        print(f"[{task.upper()}] val_breakdown:", json.dumps(val_breakdown))
    return metrics_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("cds", "tis", "all"), default="all")
    parser.add_argument("--stage3-dir", type=Path, default=DEFAULT_STAGE3)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tasks = ["cds", "tis"] if args.task == "all" else [args.task]
    for t in tasks:
        train_one_task(t, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
