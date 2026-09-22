"""
train_ner.py — fine-tune the BIO head: pretrained BERT, tqdm progress,
dev span-F1 checkpointing, TPU/GPU/CPU (docs/NER_model.md E4).

  python data_builder.py                       # first: build/ner_*.jsonl
  python train_ner.py                          # auto backend
  python train_ner.py --backend tpu            # 1 TPU core
  torchrun --nproc_per_node=8 train_ner.py     # all cores (TPU VM / GPU box)
  XLA_USE_BF16=1 python train_ner.py           # bf16 on TPU

Best epoch (by dev micro-F1) is saved to runs/<name>/ with report.json.
Skipped on purpose: AMP/fp32 is fine at this scale, grad accumulation
(batch 8 fits), DDP-CUDA (add when a multi-GPU box exists).
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

import config
from accelerator import Accelerator
from config import BIO_ID2LABEL, IGNORE_INDEX
from dataset import NerDataset, make_collator
from metrics import span_prf
from model import build_model, load_model


def decode_spans(label_ids: list, has_token: list) -> list[tuple]:
    """label-id sequence -> (start_tok, end_tok, type).
    -100 inside the attention mask is transparent (continuation sub-tokens,
    [CLS]); O breaks a run; I after O/diff-type starts fresh (defensive)."""
    spans: list[tuple] = []
    cur = None
    for pos, (tid, on) in enumerate(zip(label_ids, has_token)):
        if not on or tid == IGNORE_INDEX:
            continue
        lab = BIO_ID2LABEL[int(tid)]
        if lab == "O":
            if cur:
                spans.append(tuple(cur))
                cur = None
            continue
        begin, typ = lab.split("-", 1)
        if begin == "B" or cur is None or cur[2] != typ:
            if cur:
                spans.append(tuple(cur))
            cur = [pos, pos, typ]
        else:
            cur[1] = pos
    if cur:
        spans.append(tuple(cur))
    return spans


def evaluate(model, loader, acc: Accelerator, desc: str) -> dict:
    """No collectives inside — safe to run on rank 0 only."""
    model.eval()
    gold: list[tuple] = []
    pred: list[tuple] = []
    for batch in tqdm(loader, desc=desc, leave=False,
                      disable=not acc.is_master()):
        batch = acc.to(batch)
        with torch.no_grad():
            logits = model(**batch).logits
        pr = logits.argmax(-1)
        for i in range(pr.shape[0]):
            mask = batch["attention_mask"][i].tolist()
            gold.extend(decode_spans(batch["labels"][i].tolist(), mask))
            pred.extend(decode_spans(pr[i].tolist(), mask))
    return span_prf(gold, pred)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the NER head (BIO over BERT).")
    ap.add_argument("--model", default="bert-base-cased")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8, help="per-core batch size")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "tpu", "cuda", "cpu"])
    ap.add_argument("--run-name", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0,
                    help="keep 0 on TPU (xla + forked workers don't mix)")
    a = ap.parse_args()

    acc = Accelerator(a.backend, a.seed)
    print(f"backend: {acc.device.type}  world: {acc.world}  model: {a.model}")

    for split in ("train", "dev", "test"):
        p = config.build_file(f"ner_{split}")
        if not p.exists():
            raise SystemExit(f"missing {p} — run: python data_builder.py")

    tok = AutoTokenizer.from_pretrained(a.model)
    train_ds = NerDataset(config.build_file("ner_train"), tok, a.max_len)
    dev_ds = NerDataset(config.build_file("ner_dev"), tok, a.max_len)
    test_ds = NerDataset(config.build_file("ner_test"), tok, a.max_len)
    if not len(train_ds):
        raise SystemExit("empty train split — run: python data_builder.py")
    trunc = {"train": train_ds.truncated, "dev": dev_ds.truncated,
             "test": test_ds.truncated}
    if any(trunc.values()):
        print(f"truncated rows ({trunc}) — raise --max-len if non-zero")

    coll = make_collator(tok)
    pin = acc.device.type == "cuda"
    train_dl = DataLoader(train_ds, batch_size=a.batch, shuffle=True,
                          num_workers=a.num_workers, collate_fn=coll,
                          pin_memory=pin)
    dev_dl = DataLoader(dev_ds, batch_size=a.batch, shuffle=False,
                        num_workers=a.num_workers, collate_fn=coll,
                        pin_memory=pin)
    test_dl = DataLoader(test_ds, batch_size=a.batch, shuffle=False,
                         num_workers=a.num_workers, collate_fn=coll,
                         pin_memory=pin)

    model = build_model(a.model).to(acc.device)

    # HF canonical recipe: no weight decay on biases / layer-norms
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    opt = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters()
                    if not any(x in n for x in no_decay)],
         "weight_decay": a.weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if any(x in n for x in no_decay)],
         "weight_decay": 0.0},
    ], lr=a.lr)
    total_steps = math.ceil(len(train_dl) / acc.world) * a.epochs
    sched = get_linear_schedule_with_warmup(
        opt, int(total_steps * a.warmup_ratio), total_steps)

    run_dir = config.RUNS_DIR / (a.run_name or
                                 f"{a.model.replace('/', '-')}_{time.strftime('%Y%m%d-%H%M%S')}")
    report = {"model": a.model, "config": {k: v for k, v in vars(a).items()},
              "world": acc.world, "epochs": []}
    best = {"f1": -1.0, "epoch": -1}

    opt.zero_grad()
    for ep in range(1, a.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum, n_steps = 0.0, 0
        bar = tqdm(acc.loader(train_dl), desc=f"epoch {ep}/{a.epochs}",
                   disable=not acc.is_master())
        for batch in bar:
            batch = acc.to(batch)
            loss = model(**batch).loss
            loss.backward()
            acc.optimizer_step(opt, model.parameters(), a.clip)
            sched.step()
            loss_sum += loss.item()
            n_steps += 1
            if acc.is_master():
                bar.set_postfix(loss=f"{loss_sum / n_steps:.3f}",
                                lr=f"{sched.get_last_lr()[0]:.1e}")

        row = {"epoch": ep,
               "loss": round(loss_sum / max(n_steps, 1), 4),
               "sec": round(time.time() - t0, 1)}
        # eval has no collectives: rank 0 only, others wait at next backward
        if acc.is_master():
            dev = evaluate(model, dev_dl, acc, f"dev {ep}")
            row["dev_micro_f1"] = dev["micro"]["f1"]
            tqdm.write(f"epoch {ep}: dev micro F1 {dev['micro']['f1']:.4f}  "
                       f"loss {row['loss']}  ({row['sec']}s)")
            if dev["micro"]["f1"] > best["f1"]:
                best = {"f1": dev["micro"]["f1"], "epoch": ep}
                model.save_pretrained(run_dir)
                tok.save_pretrained(run_dir)
                (run_dir / "dev_metrics.json").write_text(
                    json.dumps(dev, indent=1), encoding="utf-8")
        report["epochs"].append(row)

    if acc.is_master():
        if best["epoch"] > 0:
            model = load_model(str(run_dir)).to(acc.device)
        test = evaluate(model, test_dl, acc, "test")
        report["best"] = best
        report["test"] = test
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "report.json").write_text(
            json.dumps(report, indent=1), encoding="utf-8")
        print(json.dumps({"best": best, "test_micro_f1": test["micro"]["f1"]},
                         indent=1))
        print(f"-> {run_dir}\n   python predict.py --model {run_dir} \"...\"")


if __name__ == "__main__":
    main()
