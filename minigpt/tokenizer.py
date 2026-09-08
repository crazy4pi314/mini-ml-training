"""A character-level tokenizer.

A **token** is simply the smallest chunk of text the model is allowed to look
at.  Big models use "sub-word" tokens (``" straw"``, ``"berry"``).  We use one
character per token because:

* the vocabulary is tiny (~60 symbols instead of 50,000), so the model is tiny;
* viewers can *see* the whole vocabulary printed on one screen;
* the model starts producing recognisable words within a couple of minutes.

Everything a tokenizer does is a lookup table in two directions:

    text  --encode-->  list of integers (token IDs)
    IDs   --decode-->  text
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

#: Characters that are always in the vocabulary, even if the corpus lacks them.
#: This keeps a checkpoint usable when we later fine-tune on a different corpus.
BASE_ALPHABET = (
    "\n !\"#'(),-.0123456789:;?[]"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
)

#: Stand-in for any character the tokenizer has never seen before.
UNK = "\x00"


class CharTokenizer:
    """Map single characters to integer IDs and back.

    Parameters
    ----------
    chars:
        The ordered vocabulary.  Use :meth:`from_text` to build it from a corpus.
    """

    def __init__(self, chars: Sequence[str]) -> None:
        # UNK always gets ID 0 so that "unknown" is easy to spot in printouts.
        vocab = [UNK] + [c for c in dict.fromkeys(chars) if c != UNK]
        self.chars: list[str] = vocab
        self.stoi: dict[str, int] = {c: i for i, c in enumerate(vocab)}
        self.itos: dict[int, str] = {i: c for i, c in enumerate(vocab)}

    # ------------------------------------------------------------------ build
    @classmethod
    def from_text(cls, text: str, extra: str = BASE_ALPHABET) -> "CharTokenizer":
        """Build a vocabulary from ``text`` plus a fixed base alphabet."""
        return cls(sorted(set(text) | set(extra)))

    # ------------------------------------------------------------------ sizes
    @property
    def vocab_size(self) -> int:
        """How many distinct tokens exist.  This is the model's output width."""
        return len(self.chars)

    def __len__(self) -> int:
        return self.vocab_size

    # --------------------------------------------------------------- encoding
    def encode(self, text: str) -> list[int]:
        """Text -> token IDs."""
        stoi = self.stoi
        return [stoi.get(ch, 0) for ch in text]

    def decode(self, ids: Iterable[int]) -> str:
        """Token IDs -> text."""
        itos = self.itos
        return "".join(itos.get(int(i), UNK) for i in ids)

    def preview(self, text: str) -> list[tuple[str, int]]:
        """``[(character, id), ...]`` - handy for a side-by-side table."""
        return [(ch, self.stoi.get(ch, 0)) for ch in text]

    # ------------------------------------------------------------------- i/o
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"chars": self.chars}, ensure_ascii=False),
            encoding="utf-8",
            newline="\n",
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload["chars"])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"CharTokenizer(vocab_size={self.vocab_size})"


__all__ = ["BASE_ALPHABET", "UNK", "CharTokenizer"]
