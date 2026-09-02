"""The training loop, evaluation, sampling, and checkpoint plumbing.

Plain-English glossary for the words that show up here:

**loss**
    A single number that says "how surprised was the model by the correct next
    token?"  Lower is better.  Because we use cross-entropy on a vocabulary of
    ~64 characters, a model that has learned *nothing* scores about
    ``ln(64) = 4.16``.  Anything below that means it has learned something.

**step (or iteration)**
    One peek at one batch of examples, followed by one small nudge to every
    parameter.

**epoch**
    One full pass over the whole training set.  We usually count steps instead,
    because our dataset is small enough that we'd blow through many epochs.

**batch size**
    How many chunks of text we look at before nudging the parameters.  Bigger
    batches give a steadier, less jumpy estimate of which way to nudge.

**learning rate**
    How big each nudge is.  Too small and the model crawls (underfitting); too
    big and it thrashes around and the loss goes haywire.

**checkpoint**
    A saved copy of every parameter, so you can stop and pick up later - or hand
    the model to someone else.

**validation loss**
    The same loss, measured on text the model has never trained on.  This is the
    only number that tells you whether it *learned* or merely *memorised*.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

from minigpt.model import MiniGPT, MiniGPTConfig
from minigpt.paths import ARTIFACTS, CHECKPOINTS
from minigpt.tokenizer import CharTokenizer

DEVICE = torch.device("cpu")  # this whole repo is deliberately CPU-only


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int = 1337) -> None:
    """Pin every random number generator so a run is repeatable.

    Live demos are much less stressful when the output is the same every time.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)  # not needed on CPU, and faster off


# ---------------------------------------------------------------------------
# Turning text into batches
# ---------------------------------------------------------------------------


def encode_to_tensor(text: str, tokenizer: CharTokenizer) -> torch.Tensor:
    """Whole corpus -> one long 1-D tensor of token IDs."""
    return torch.tensor(tokenizer.encode(text), dtype=torch.long)


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grab ``batch_size`` random windows of ``block_size`` tokens.

    ``y`` is just ``x`` shifted one token to the left: at every position the
    model's job is "predict the next character".
    """
    high = max(1, len(data) - block_size - 1)
    ix = torch.randint(high, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x, y


@torch.no_grad()
def evaluate(
    model: MiniGPT,
    data: torch.Tensor,
    block_size: int,
    batch_size: int = 32,
    iters: int = 20,
    seed: int = 0,
) -> float:
    """Average loss over ``iters`` random batches, with dropout switched off."""
    was_training = model.training
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    total = 0.0
    for _ in range(iters):
        x, y = get_batch(data, block_size, batch_size, gen)
        _, loss = model(x, y)
        total += float(loss)
    if was_training:
        model.train()
    return total / iters


# ---------------------------------------------------------------------------
# Recording what happened, so we can plot it later
# ---------------------------------------------------------------------------


@dataclass
class History:
    """The record of a training run - everything the charts in 03 need."""

    name: str = "run"
    steps: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    samples: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def record(self, step: int, train_loss: float, val_loss: float) -> None:
        self.steps.append(int(step))
        self.train_loss.append(float(train_loss))
        self.val_loss.append(float(val_loss))

    def add_sample(self, step: int, text: str) -> None:
        self.samples.append({"step": int(step), "text": text})

    @property
    def best_val(self) -> float:
        return min(self.val_loss) if self.val_loss else float("nan")

    @property
    def best_val_step(self) -> int:
        if not self.val_loss:
            return -1
        return self.steps[int(np.argmin(self.val_loss))]

    @property
    def final_gap(self) -> float:
        """Validation loss minus training loss at the end of the run.

        A big positive gap is the classic fingerprint of overfitting.
        """
        if not self.val_loss:
            return float("nan")
        return self.val_loss[-1] - self.train_loss[-1]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "steps": self.steps,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "samples": self.samples,
            "meta": self.meta,
        }


def save_history(history: History, path: str | Path | None = None) -> Path:
    """Write a run history to ``artifacts/<name>.json``."""
    path = Path(path) if path else ARTIFACTS / f"{history.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history.to_dict(), indent=2), encoding="utf-8")
    return path


def load_history(name_or_path: str | Path) -> History:
    """Read back a run history saved by :func:`save_history`."""
    path = Path(name_or_path)
    if not path.suffix:
        path = ARTIFACTS / f"{path.name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `python scripts/pretrain_all.py` to bake the artifacts."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return History(**payload)


# ---------------------------------------------------------------------------
# The training loop itself
# ---------------------------------------------------------------------------


def train_model(
    model: MiniGPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    *,
    steps: int = 1500,
    batch_size: int = 32,
    learning_rate: float = 3e-3,
    weight_decay: float = 0.01,
    eval_every: int = 100,
    eval_iters: int = 20,
    eval_batch_size: int = 32,
    warmup_steps: int = 50,
    min_lr_ratio: float = 0.1,
    cosine_decay: bool = True,
    grad_clip: float = 1.0,
    seed: int = 1337,
    name: str = "run",
    on_eval: Callable[[int, float, float], None] | None = None,
    verbose: bool = True,
    history: History | None = None,
) -> History:
    """Train ``model`` and return the :class:`History` of the run.

    Every argument that a beginner would ask about:

    ``steps``
        How many nudges to make in total.
    ``batch_size``
        How many text windows go into each nudge.
    ``learning_rate``
        How big each nudge is.
    ``weight_decay``
        A gentle pull of every parameter back towards zero; another way of
        discouraging memorisation.
    ``eval_every``
        How often to stop and take the pop quiz (measure validation loss).
    ``warmup_steps`` / ``cosine_decay``
        Start with tiny nudges, grow to the full learning rate, then ease off
        towards the end.  This is standard practice and makes runs much stabler.
    ``grad_clip``
        Cap on how large a single nudge can be, so one weird batch cannot
        destroy the model.
    """
    set_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    block_size = model.cfg.block_size
    history = history or History(name=name)
    history.meta.update(
        {
            "steps": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "seed": seed,
            "n_params": model.num_params(),
            "block_size": block_size,
            "train_tokens": int(len(train_data)),
            "val_tokens": int(len(val_data)),
            **model.cfg.to_dict(),
        }
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.99)
    )

    def lr_at(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return learning_rate * (step + 1) / warmup_steps
        if not cosine_decay:
            return learning_rate
        progress = (step - warmup_steps) / max(1, steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        scale = min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        return learning_rate * scale

    model.train()
    started = time.time()

    for step in range(steps + 1):
        if step % eval_every == 0 or step == steps:
            tr = evaluate(model, train_data, block_size, eval_batch_size, eval_iters, seed=step)
            va = evaluate(model, val_data, block_size, eval_batch_size, eval_iters, seed=step + 1)
            history.record(step, tr, va)
            if verbose:
                print(
                    f"  step {step:>5} | train loss {tr:.4f} | val loss {va:.4f} "
                    f"| {time.time() - started:5.1f}s"
                )
            if on_eval is not None:
                on_eval(step, tr, va)
        if step == steps:
            break

        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)

        x, y = get_batch(train_data, block_size, batch_size, gen)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    history.meta["wall_seconds"] = round(time.time() - started, 2)
    history.meta["final_train_loss"] = history.train_loss[-1]
    history.meta["final_val_loss"] = history.val_loss[-1]
    if verbose:
        print(f"  done in {history.meta['wall_seconds']}s")
    return history


# ---------------------------------------------------------------------------
# Generating text
# ---------------------------------------------------------------------------


def generate(
    model: MiniGPT,
    tokenizer: CharTokenizer,
    prompt: str = "\n",
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = 40,
    seed: int | None = 1234,
) -> str:
    """Ask the model to continue ``prompt`` and return prompt + continuation."""
    if seed is not None:
        torch.manual_seed(seed)
    was_training = model.training
    ids = tokenizer.encode(prompt) or [0]
    idx = torch.tensor([ids[-model.cfg.block_size :]], dtype=torch.long)
    out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
    if was_training:
        model.train()
    return tokenizer.decode(out[0].tolist())


def show_sample(
    model: MiniGPT,
    tokenizer: CharTokenizer,
    prompt: str = "\n",
    max_new_tokens: int = 180,
    label: str = "",
    **kwargs,
) -> str:
    """Print a generation inside a labelled box (nice on a stream)."""
    text = generate(model, tokenizer, prompt, max_new_tokens, **kwargs)
    bar = "-" * 72
    print(bar)
    if label:
        print(label)
        print(bar)
    print(text)
    print(bar)
    return text


def ask(
    model: MiniGPT,
    tokenizer: CharTokenizer,
    question: str,
    max_new_tokens: int = 60,
    temperature: float = 0.5,
    top_k: int | None = 10,
    seed: int | None = 1234,
) -> str:
    """Ask the model a question in the fine-tuning format and read the answer.

    Generation stops at the first newline, because that is the "I am finished"
    signal the model was fine-tuned to emit.  A model that has *not* been
    fine-tuned will just ramble until it hits ``max_new_tokens`` - which is
    exactly the before/after contrast we want in notebook 05.
    """
    from minigpt.data import PROMPT_TEMPLATE

    prompt = PROMPT_TEMPLATE.format(question=question)
    text = generate(model, tokenizer, prompt, max_new_tokens, temperature, top_k, seed)
    completion = text[len(prompt) :]
    return completion.split("\n")[0].strip()


def compare_answers(
    before: MiniGPT,
    after: MiniGPT,
    tokenizer: CharTokenizer,
    questions: Sequence[str],
    **kwargs,
) -> None:
    """Print the same questions answered by two models, side by side."""
    for question in questions:
        print(f"Q: {question}")
        print(f"   base model : {ask(before, tokenizer, question, **kwargs)!r}")
        print(f"   fine-tuned : {ask(after, tokenizer, question, **kwargs)!r}")
        print()


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: MiniGPT,
    tokenizer: CharTokenizer,
    name: str,
    extra: dict | None = None,
) -> Path:
    """Save weights + config + vocabulary into ``checkpoints/<name>.pt``."""
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINTS / f"{name}.pt"
    torch.save(
        {
            "config": model.cfg.to_dict(),
            "state_dict": model.state_dict(),
            "vocab": tokenizer.chars,
            "extra": extra or {},
        },
        path,
    )
    return path


def load_checkpoint(name_or_path: str | Path) -> tuple[MiniGPT, CharTokenizer, dict]:
    """Load a checkpoint back into ``(model, tokenizer, extra)``."""
    path = Path(name_or_path)
    if not path.suffix:
        path = CHECKPOINTS / f"{path.name}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `python scripts/pretrain_all.py` (or notebook 02) first."
        )
    payload = torch.load(path, map_location=DEVICE, weights_only=False)
    model = MiniGPT(MiniGPTConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    tokenizer = CharTokenizer(payload["vocab"])
    return model, tokenizer, payload.get("extra", {})


# ---------------------------------------------------------------------------
# Supervised fine-tuning (notebook 05)
# ---------------------------------------------------------------------------


def build_sft_batches(
    pairs: Sequence[dict],
    tokenizer: CharTokenizer,
    block_size: int,
    mask_prompt: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn question/answer pairs into ``(inputs, targets)`` tensors.

    **Loss masking** is the important idea here.  Each example is the prompt
    glued to the answer, but we set the target to ``-100`` on every prompt
    position, and our model's loss ignores ``-100``.  So the model is graded
    *only* on the answer.  Without this it would spend most of its effort
    learning to write questions, which is not the job we are hiring it for.
    """
    from minigpt.data import format_example

    xs, ys = [], []
    for pair in pairs:
        prompt, answer = format_example(pair["question"], pair["answer"])
        p_ids = tokenizer.encode(prompt)
        a_ids = tokenizer.encode(answer)
        ids = (p_ids + a_ids)[: block_size + 1]
        if len(ids) < 8:
            continue
        pad = block_size + 1 - len(ids)
        ids = ids + [0] * pad

        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        if mask_prompt:
            # Positions 0..len(prompt)-2 predict prompt tokens -> ignore them.
            y[: max(0, len(p_ids) - 1)] = -100
        if pad:
            y[-pad:] = -100  # never grade the padding either
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)


def train_sft(
    model: MiniGPT,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    val_x: torch.Tensor | None = None,
    val_y: torch.Tensor | None = None,
    steps: int = 400,
    batch_size: int = 32,
    learning_rate: float = 5e-4,
    eval_every: int = 50,
    seed: int = 1337,
    name: str = "sft",
    verbose: bool = True,
) -> History:
    """Fine-tune on pre-batched, loss-masked instruction data."""
    set_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    history = History(name=name)
    history.meta.update(
        {"steps": steps, "batch_size": batch_size, "learning_rate": learning_rate,
         "n_examples": int(x.shape[0]), "seed": seed}
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    started = time.time()

    @torch.no_grad()
    def masked_eval(xx: torch.Tensor, yy: torch.Tensor, iters: int = 8) -> float:
        model.eval()
        g = torch.Generator().manual_seed(0)
        total = 0.0
        for _ in range(iters):
            ix = torch.randint(xx.shape[0], (min(batch_size, xx.shape[0]),), generator=g)
            _, loss = model(xx[ix], yy[ix])
            total += float(loss)
        model.train()
        return total / iters

    model.train()
    for step in range(steps + 1):
        if step % eval_every == 0 or step == steps:
            tr = masked_eval(x, y)
            va = masked_eval(val_x, val_y) if val_x is not None else tr
            history.record(step, tr, va)
            if verbose:
                print(f"  step {step:>5} | train {tr:.4f} | val {va:.4f} "
                      f"| {time.time() - started:5.1f}s")
        if step == steps:
            break
        ix = torch.randint(x.shape[0], (batch_size,), generator=gen)
        _, loss = model(x[ix], y[ix])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    history.meta["wall_seconds"] = round(time.time() - started, 2)
    return history


# ---------------------------------------------------------------------------
# Helpers for continued pre-training (notebook 04)
# ---------------------------------------------------------------------------


def mix_texts(new_text: str, old_text: str, replay_fraction: float = 0.3, seed: int = 0) -> str:
    """Blend a slice of the *original* corpus back into the *new* corpus.

    This is called **replay** (or rehearsal).  It is the cheapest known fix for
    catastrophic forgetting: while the model learns the new domain, keep showing
    it a little of the old one so it does not paper over what it already knew.
    """
    if not 0.0 <= replay_fraction < 1.0:
        raise ValueError("replay_fraction must be in [0, 1)")
    if replay_fraction == 0.0:
        return new_text
    n_old = int(len(new_text) * replay_fraction / (1 - replay_fraction))
    old_docs = [d for d in old_text.split("\n\n") if d.strip()]
    rng = random.Random(seed)
    rng.shuffle(old_docs)
    picked, size = [], 0
    for doc in old_docs:
        if size >= n_old:
            break
        picked.append(doc)
        size += len(doc) + 2
    new_docs = [d for d in new_text.split("\n\n") if d.strip()]
    combined = new_docs + picked
    rng.shuffle(combined)
    return "\n\n".join(combined) + "\n"


def memorization_score(
    model: MiniGPT,
    tokenizer: CharTokenizer,
    passage: str,
    prefix_chars: int = 60,
    temperature: float = 0.2,
) -> dict:
    """How much of ``passage`` does the model recite when given its opening?

    Returns the continuation plus the length of the longest matching prefix,
    which is a blunt but very legible measure of verbatim memorisation.
    """
    prefix = passage[:prefix_chars]
    truth = passage[prefix_chars:]
    produced = generate(
        model, tokenizer, prefix, max_new_tokens=len(truth), temperature=temperature, top_k=5
    )[len(prefix) :]
    match = 0
    for a, b in zip(truth, produced):
        if a != b:
            break
        match += 1
    return {
        "prefix": prefix,
        "truth": truth,
        "produced": produced,
        "matching_chars": match,
        "matching_fraction": match / max(1, len(truth)),
    }


def preview_loss_mask(
    pair: dict, tokenizer: CharTokenizer, block_size: int, width: int = 88
) -> None:
    """Print one fine-tuning example with the graded characters marked.

    ``.`` = position ignored by the loss (the prompt and the padding)
    ``^`` = position the model is actually graded on (the answer)
    """
    from minigpt.data import format_example

    prompt, answer = format_example(pair["question"], pair["answer"])
    x, y = build_sft_batches([pair], tokenizer, block_size)
    chars = [tokenizer.decode([i]) for i in x[0].tolist()]
    graded = [t != -100 for t in y[0].tolist()]

    shown = "".join("\\n" if c == "\n" else ("_" if c == "\x00" else c) for c in chars)
    marks = "".join(
        ("^" if g else ".") * (2 if c == "\n" else 1) for c, g in zip(chars, graded)
    )
    for start in range(0, len(shown), width):
        print(shown[start : start + width])
        print(marks[start : start + width])
    n_graded = sum(graded)
    print(
        f"\n{n_graded} of {len(graded)} positions are graded "
        f"({n_graded / len(graded):.0%}) - the prompt and the padding are free."
    )


__all__ = [
    "DEVICE",
    "History",
    "ask",
    "build_sft_batches",
    "compare_answers",
    "encode_to_tensor",
    "evaluate",
    "generate",
    "get_batch",
    "load_checkpoint",
    "load_history",
    "memorization_score",
    "mix_texts",
    "preview_loss_mask",
    "save_checkpoint",
    "save_history",
    "set_seed",
    "show_sample",
    "train_model",
    "train_sft",
]
