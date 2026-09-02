"""A very small GPT, written out in full so it fits on one screen at a time.

This is a trimmed-down cousin of Andrej Karpathy's nanoGPT.  The whole model is
about 0.8 million parameters - roughly 1/200,000th the size of a frontier model
- which is exactly why it trains on a laptop CPU in a couple of minutes.

Plain-English glossary for the pieces below:

**parameter**
    A single number the model is allowed to adjust while it learns.  "0.8M
    parameters" means 800,000 adjustable dials.

**embedding**
    A lookup table that turns a token ID into a list of numbers, because neural
    networks do arithmetic, not spelling.

**attention**
    The step where every position in the text gets to look back at the earlier
    positions and decide which ones matter for predicting the next token.

**block / layer**
    One round of "look back at the context, then think about it".  Stacking
    several blocks lets the model build up more complicated patterns.

**logits**
    The raw, unnormalised scores the model gives to every possible next token.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class MiniGPTConfig:
    """Every knob that defines the shape of the model.

    Attributes
    ----------
    vocab_size:
        How many distinct tokens exist (for us: how many characters).
    block_size:
        The **context window** - how many tokens the model may look back at.
    n_layer:
        How many transformer blocks are stacked on top of each other (depth).
    n_head:
        How many attention "heads" run in parallel inside each block.  More
        heads = more independent things the model can pay attention to at once.
    n_embd:
        The width of the model - the length of the number-list used to
        represent each token.  Must divide evenly by ``n_head``.
    dropout:
        During training, randomly zero out this fraction of internal values.
        It is a cheap way to stop the model from memorising the training data.
    """

    vocab_size: int = 64
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    """Let each position look back at (only) the positions before it."""

    def __init__(self, cfg: MiniGPTConfig) -> None:
        super().__init__()
        if cfg.n_embd % cfg.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout

        # One matrix produces the query/key/value triples for every head at once.
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # The "causal mask": position t may not peek at positions > t.
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
            1, 1, cfg.block_size, cfg.block_size
        )
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, time (tokens), channels (n_embd)
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        # Split the channel dimension across the heads: (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # How much should each position care about each earlier position?
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v  # weighted average of the values
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-merge the heads
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    """The "think about it" half of a block: a small 2-layer network."""

    def __init__(self, cfg: MiniGPTConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """attention (look around) + MLP (think), each added back onto the input."""

    def __init__(self, cfg: MiniGPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """Predict the next token, given up to ``block_size`` previous tokens."""

    def __init__(self, cfg: MiniGPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying: the "read a token in" and "write a token out" tables are
        # the same table.  Fewer parameters, and it trains a little faster.
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------ stats
    def num_params(self) -> int:
        """Total number of adjustable numbers in the model."""
        return sum(p.numel() for p in self.parameters())

    # ---------------------------------------------------------------- forward
    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the model.

        Parameters
        ----------
        idx:
            ``(batch, time)`` tensor of token IDs.
        targets:
            Optional ``(batch, time)`` tensor of "the token that actually came
            next".  Positions set to ``-100`` are **ignored** by the loss - that
            is how we do *loss masking* during supervised fine-tuning.

        Returns
        -------
        ``(logits, loss)`` where ``loss`` is ``None`` when no targets are given.
        """
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(
                f"sequence of {T} tokens is longer than block_size={self.cfg.block_size}"
            )
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    # --------------------------------------------------------------- sampling
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.9,
        top_k: int | None = 40,
    ) -> torch.Tensor:
        """Write ``max_new_tokens`` more tokens, one at a time.

        ``temperature`` controls how adventurous the model is: values below 1
        make it play it safe, values above 1 make it wilder.  ``top_k`` throws
        away everything except the ``k`` most likely next tokens.
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                cutoff = torch.topk(logits, k)[0][..., -1, None]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


#: Batch size that pairs with :func:`default_config` on a small CPU.
DEFAULT_BATCH_SIZE = 12


def default_config(vocab_size: int, **overrides) -> MiniGPTConfig:
    """The house model: ~165k parameters, sized to train in ~2 minutes on a CPU.

    ``block_size=96`` is chosen so that a whole instruction-tuning example
    (question **and** answer) fits inside one context window in notebook 05.

    Notebooks and ``scripts/pretrain_all.py`` both call this so a checkpoint
    baked ahead of time always matches the model the notebook builds live.
    """
    cfg = MiniGPTConfig(
        vocab_size=vocab_size,
        block_size=96,
        n_layer=3,
        n_head=4,
        n_embd=64,
        dropout=0.1,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "Block",
    "CausalSelfAttention",
    "MLP",
    "MiniGPT",
    "MiniGPTConfig",
    "default_config",
]
