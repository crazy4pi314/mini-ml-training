"""Bake every checkpoint and loss history ahead of the live stream.

    python scripts/pretrain_all.py            # everything (~6-8 minutes on a CPU)
    python scripts/pretrain_all.py --only healthy overfit

Why this exists
---------------
Notebook 03 needs three *real* training runs to point at - an underfit one, a
healthy one, and an overfit one - and notebooks 04/05 need a base checkpoint to
start from.  Training those live would eat the whole hour, so we do it once,
save the results as JSON + ``.pt`` files, and let every notebook flip
``USE_PREBAKED = True`` to load them instantly.

Everything here is seeded, so re-running reproduces the same numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from minigpt import data, train as T  # noqa: E402
from minigpt.model import DEFAULT_BATCH_SIZE, MiniGPT, default_config  # noqa: E402
from minigpt.paths import ARTIFACTS, CHECKPOINTS, DATA, ensure_dirs  # noqa: E402

SEED = 1337

# Step budgets, chosen so no single run takes longer than ~2 minutes on a
# modest CPU.  Lower them if you are on a very small Codespace.
HEALTHY_STEPS = 1200
UNDERFIT_STEPS = 250
OVERFIT_STEPS = 1200
CPT_STEPS = 400
SFT_STEPS = 400

SAMPLE_PROMPT = "\n"


def _tensors(tokenizer):
    """Load every split we need as token-ID tensors."""
    base_train = data.load_split("base", "train")
    base_val = data.load_split("base", "val")
    rec_train = data.load_split("recipes", "train")
    rec_val = data.load_split("recipes", "val")
    tiny_train = (DATA / "tiny_train.txt").read_text(encoding="utf-8")
    enc = lambda t: T.encode_to_tensor(t, tokenizer)  # noqa: E731
    return {
        "base_train_text": base_train,
        "base_train": enc(base_train),
        "base_val": enc(base_val),
        "rec_train_text": rec_train,
        "rec_train": enc(rec_train),
        "rec_val": enc(rec_val),
        "tiny_train": enc(tiny_train),
    }


# ---------------------------------------------------------------------------
# The three diagnosis runs
# ---------------------------------------------------------------------------


def run_healthy(tok, ts) -> T.History:
    """A well-sized model, a sensible learning rate, enough steps."""
    print("\n[healthy] the run we actually want")
    T.set_seed(SEED)
    model = MiniGPT(default_config(tok.vocab_size))
    history = T.History(name="healthy")

    def snapshot(step, _tr, _va):
        if step in (0, HEALTHY_STEPS // 2, HEALTHY_STEPS):
            history.add_sample(step, T.generate(model, tok, SAMPLE_PROMPT, 220, seed=SEED))

    T.train_model(
        model, ts["base_train"], ts["base_val"],
        steps=HEALTHY_STEPS, batch_size=DEFAULT_BATCH_SIZE, learning_rate=3e-3,
        eval_every=100, eval_iters=16, eval_batch_size=32,
        seed=SEED, name="healthy", history=history, on_eval=snapshot,
    )
    history.meta["diagnosis"] = "generalising: both curves fall together"
    T.save_checkpoint(model, tok, "base", extra={"history": "healthy"})
    T.save_history(history)
    return history


def run_underfit(tok, ts) -> T.History:
    """Same model, but the learning rate is tiny and we stop far too early."""
    print("\n[underfit] learning rate 30x too small, stopped 5x too early")
    T.set_seed(SEED)
    model = MiniGPT(default_config(tok.vocab_size))
    history = T.History(name="underfit")
    history.add_sample(0, T.generate(model, tok, SAMPLE_PROMPT, 200, seed=SEED))
    T.train_model(
        model, ts["base_train"], ts["base_val"],
        steps=UNDERFIT_STEPS, batch_size=DEFAULT_BATCH_SIZE, learning_rate=1e-4,
        eval_every=25, eval_iters=16, eval_batch_size=32, warmup_steps=10,
        seed=SEED, name="underfit", history=history,
    )
    history.add_sample(UNDERFIT_STEPS, T.generate(model, tok, SAMPLE_PROMPT, 200, seed=SEED))
    history.meta["diagnosis"] = "underfitting: both curves are still high and still falling"
    T.save_history(history)
    return history


def run_overfit(tok, ts) -> T.History:
    """Same model, a *tiny* training set, no dropout, and far too many passes."""
    print("\n[overfit] 6 documents, no dropout, 1200 steps")
    T.set_seed(SEED)
    model = MiniGPT(default_config(tok.vocab_size, dropout=0.0))
    history = T.History(name="overfit")
    T.train_model(
        model, ts["tiny_train"], ts["base_val"],
        steps=OVERFIT_STEPS, batch_size=DEFAULT_BATCH_SIZE, learning_rate=3e-3,
        weight_decay=0.0, eval_every=75, eval_iters=16, eval_batch_size=32,
        cosine_decay=False, seed=SEED, name="overfit", history=history,
    )
    history.meta["diagnosis"] = "overfitting: train keeps falling, validation turns back up"
    T.save_checkpoint(model, tok, "overfit", extra={"history": "overfit"})
    T.save_history(history)
    return history


# ---------------------------------------------------------------------------
# Continued pre-training (notebook 04)
# ---------------------------------------------------------------------------


def _run_cpt(tok, ts, name: str, learning_rate: float, replay: float) -> T.History:
    model, _tokenizer, _extra = T.load_checkpoint("base")
    history = T.History(name=name)

    if replay > 0:
        text = T.mix_texts(ts["rec_train_text"], ts["base_train_text"], replay, seed=SEED)
        train_data = T.encode_to_tensor(text, tok)
    else:
        train_data = ts["rec_train"]

    old_val: list[float] = []

    def track_old_domain(step, _tr, _va):
        old_val.append(
            T.evaluate(model, ts["base_val"], model.cfg.block_size, 32, 16, seed=step + 2)
        )

    T.train_model(
        model, train_data, ts["rec_val"],
        steps=CPT_STEPS, batch_size=DEFAULT_BATCH_SIZE, learning_rate=learning_rate,
        eval_every=50, eval_iters=16, eval_batch_size=32, warmup_steps=20,
        seed=SEED, name=name, history=history, on_eval=track_old_domain,
    )
    history.meta["old_domain_val"] = old_val
    history.meta["replay_fraction"] = replay
    history.add_sample(CPT_STEPS, T.generate(model, tok, "recipe:", 220, seed=SEED))
    history.add_sample(-1, T.generate(model, tok, "one day ", 220, seed=SEED))
    T.save_checkpoint(model, tok, name, extra={"history": name})
    T.save_history(history)
    return history


def run_cpt_naive(tok, ts) -> T.History:
    print("\n[cpt-naive] new domain only, full learning rate -> forgetting")
    return _run_cpt(tok, ts, "cpt_naive", learning_rate=3e-3, replay=0.0)


def run_cpt_replay(tok, ts) -> T.History:
    print("\n[cpt-replay] lower learning rate + 30% replay of the old corpus")
    return _run_cpt(tok, ts, "cpt_replay", learning_rate=5e-4, replay=0.3)


# ---------------------------------------------------------------------------
# Supervised fine-tuning (notebook 05)
# ---------------------------------------------------------------------------


def run_sft(tok, _ts) -> T.History:
    print("\n[sft] instruction tuning on question -> answer pairs")
    model, _tokenizer, _extra = T.load_checkpoint("base")
    train_pairs = data.load_qa("sft")
    val_pairs = data.load_qa("sft_val")
    x, y = T.build_sft_batches(train_pairs, tok, model.cfg.block_size)
    vx, vy = T.build_sft_batches(val_pairs, tok, model.cfg.block_size)
    history = T.train_sft(
        model, x, y, val_x=vx, val_y=vy,
        steps=SFT_STEPS, batch_size=DEFAULT_BATCH_SIZE, learning_rate=1e-3,
        eval_every=50, seed=SEED, name="sft",
    )
    prompt = val_pairs[0]["question"]
    history.add_sample(SFT_STEPS, T.ask(model, tok, prompt, seed=SEED))
    history.meta["sample_question"] = prompt
    T.save_checkpoint(model, tok, "sft", extra={"history": "sft"})
    T.save_history(history)
    return history


# ---------------------------------------------------------------------------

RUNS = {
    "healthy": run_healthy,
    "underfit": run_underfit,
    "overfit": run_overfit,
    "cpt_naive": run_cpt_naive,
    "cpt_replay": run_cpt_replay,
    "sft": run_sft,
}

#: ``healthy`` must run first - everything after it starts from its checkpoint.
ORDER = ["healthy", "underfit", "overfit", "cpt_naive", "cpt_replay", "sft"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", choices=ORDER, help="run a subset")
    parser.add_argument("--threads", type=int, default=4, help="torch CPU threads")
    parser.add_argument("--skip-data", action="store_true", help="reuse data/ as-is")
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.threads))
    ensure_dirs()

    if not args.skip_data or not (DATA / "vocab.json").exists():
        print("[data] preparing corpora")
        data.prepare_all()

    tokenizer = data.load_tokenizer()
    ts = _tensors(tokenizer)
    print(f"[data] vocab={tokenizer.vocab_size} "
          f"train={len(ts['base_train'])} val={len(ts['base_val'])} tokens")

    wanted = args.only or ORDER
    if "healthy" not in wanted and not (CHECKPOINTS / "base.pt").exists():
        raise SystemExit("run `--only healthy` first: the other runs need checkpoints/base.pt")

    timings: dict[str, float] = {}
    overall = time.time()
    for name in ORDER:
        if name not in wanted:
            continue
        started = time.time()
        history = RUNS[name](tokenizer, ts)
        timings[name] = round(time.time() - started, 1)
        print(f"[{name}] {timings[name]}s | final train {history.train_loss[-1]:.4f} "
              f"| final val {history.val_loss[-1]:.4f}")

    report = {
        "seconds_per_run": timings,
        "total_seconds": round(time.time() - overall, 1),
        "torch": torch.__version__,
        "threads": args.threads,
    }
    (ARTIFACTS / "bake_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nAll done in {report['total_seconds']}s")
    print(f"  checkpoints -> {CHECKPOINTS}")
    print(f"  histories   -> {ARTIFACTS}")


if __name__ == "__main__":
    main()
