"""Everything to do with the *text* we train on.

The story this module tells, in order:

1. :func:`load_base_corpus` gives us a small, clean, public-domain-safe corpus.
2. :func:`make_messy_corpus` deliberately wrecks it the way the real internet
   wrecks text: duplicates, near-duplicates, mojibake, boilerplate, junk.
3. The ``drop_*``/``dedup_*``/``filter_*`` functions clean it back up, one
   understandable step at a time, reporting how many documents each step
   removed so we can show a before/after table.
4. :func:`split_train_val` carves off the **validation set** - the pop quiz the
   model is never allowed to study.

There is deliberately **no large download here**.  The default corpus is
generated from a seeded template grammar, so it is byte-for-byte identical on
every machine and needs no network at all.  ``load_base_corpus("shakespeare")``
will fetch the ~1 MB public-domain "tiny shakespeare" file if you want a real
literary corpus and have a network connection.
"""

from __future__ import annotations

import json
import random
import re
import unicodedata
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from minigpt.paths import DATA
from minigpt.tokenizer import BASE_ALPHABET

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

ALLOWED_CHARS = set(BASE_ALPHABET)


# ---------------------------------------------------------------------------
# 1. The clean source corpora
# ---------------------------------------------------------------------------

_NAMES = [
    "mila", "toby", "juno", "ravi", "nell", "otto", "wren", "hugo",
    "ida", "pip", "sana", "cleo", "bo", "lena", "finn", "roux",
]
_ANIMALS = [
    "fox", "otter", "sparrow", "badger", "turtle", "moth", "goat", "crane",
    "hedgehog", "seal", "magpie", "donkey",
]
_PLACES = [
    "old mill", "salt marsh", "pine ridge", "harbour wall", "quiet library",
    "night market", "green ferry", "clock tower", "back garden", "long beach",
]
_OBJECTS = [
    "brass key", "paper boat", "cracked bell", "blue lantern", "wool hat",
    "small map", "glass marble", "tin whistle", "silver coin", "folded note",
]
_ADJECTIVES = [
    "quiet", "bright", "patient", "curious", "sleepy", "careful", "brave",
    "gentle", "restless", "cheerful",
]
_WEATHER = [
    "the rain stopped", "the fog rolled in", "the sun came out",
    "the wind turned cold", "the sky went pink", "a light snow began",
]

_STORY_TEMPLATES = [
    "{name} lived near the {place}.",
    "{name} was a {adj} child who liked to walk by the {place}.",
    "every morning {name} went down to the {place}.",
    "one day {name} found a {obj} under a bench.",
    "the {obj} was older than it looked.",
    "{name} showed the {obj} to a {adj} {animal}.",
    "the {animal} did not say a word, but it followed {name} home.",
    "{weather}, and {name} decided to go back to the {place}.",
    "{name} asked the {animal} what to do with the {obj}.",
    "the {animal} tipped its head, which {name} took to mean yes.",
    "they walked past the {place} until the light went soft.",
    "{name} kept the {obj} in a pocket for {number} days.",
    "on the {number}th day, {name} gave the {obj} away.",
    "the {adj} {animal} came back the next week, and the week after that.",
    "nobody in the town believed the story about the {obj}.",
    "{name} did not mind, because {name} knew what happened.",
    "in the end, {name} and the {animal} shared the {obj}.",
    "that is how {name} learned to be {adj}.",
]

_RECIPE_TEMPLATES = [
    "recipe: {adj} {veg} {dish}.",
    "serves {number}. total time: {number}0 minutes.",
    "ingredients: {number} cups of {veg}, two spoons of {fat}, a pinch of salt.",
    "you will also need {number} {veg} and a little {herb}.",
    "step {number}: heat the {fat} in a wide pan over medium heat.",
    "step {number}: add the {veg} and cook until soft, about {number} minutes.",
    "step {number}: stir in the {herb} and season with salt and pepper.",
    "step {number}: pour in the stock and let it simmer.",
    "step {number}: taste it. if it is flat, add more salt.",
    "serve the {dish} warm with bread.",
    "leftovers keep in the fridge for {number} days.",
    "notes: do not let the {fat} smoke.",
    "notes: chopping the {veg} small makes the {dish} cook faster.",
    "chef tip: the {herb} goes in at the end, never at the start.",
]

_VEG = ["onion", "carrot", "leek", "potato", "squash", "bean", "tomato", "pea"]
_FAT = ["butter", "olive oil", "duck fat", "sesame oil"]
_HERB = ["thyme", "parsley", "dill", "sage", "basil", "chive"]
_DISH = ["soup", "stew", "hash", "pie", "broth", "bake", "mash"]


def _fill(rng: random.Random, template: str) -> str:
    return template.format(
        name=rng.choice(_NAMES),
        animal=rng.choice(_ANIMALS),
        place=rng.choice(_PLACES),
        obj=rng.choice(_OBJECTS),
        adj=rng.choice(_ADJECTIVES),
        weather=rng.choice(_WEATHER),
        number=rng.randint(2, 9),
        veg=rng.choice(_VEG),
        fat=rng.choice(_FAT),
        herb=rng.choice(_HERB),
        dish=rng.choice(_DISH),
    )


def _generate_documents(
    templates: Sequence[str],
    n_docs: int,
    seed: int,
    sentences: tuple[int, int] = (4, 7),
) -> list[str]:
    """Build ``n_docs`` short paragraphs from a template grammar."""
    rng = random.Random(seed)
    docs: list[str] = []
    for _ in range(n_docs):
        k = rng.randint(*sentences)
        picks = rng.sample(range(len(templates)), k=min(k, len(templates)))
        picks.sort()
        docs.append(" ".join(_fill(rng, templates[i]) for i in picks))
    return docs


def tinytales_documents(n_docs: int = 1400, seed: int = 1337) -> list[str]:
    """The default *base* corpus: gentle, repetitive little stories.

    Repetitive on purpose - a 0.8M-parameter character model needs a corpus
    with real structure if it is going to produce readable English inside a
    two-minute training run on a laptop CPU.
    """
    return _generate_documents(_STORY_TEMPLATES, n_docs, seed)


def recipe_documents(n_docs: int = 500, seed: int = 4242) -> list[str]:
    """The *new domain* corpus used in ``04_continued_pretraining``.

    Deliberately a big stylistic jump from the stories: numbered steps,
    imperative voice, cooking vocabulary the base model has never seen.
    """
    return _generate_documents(_RECIPE_TEMPLATES, n_docs, seed, sentences=(5, 9))


def fetch_tiny_shakespeare(timeout: float = 10.0) -> str | None:
    """Download the public-domain "tiny shakespeare" corpus (~1 MB).

    Returns ``None`` (instead of raising) when there is no network, so the
    notebooks never die on a flaky conference wifi connection.
    """
    cache = DATA / "tinyshakespeare.txt"
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    try:  # pragma: no cover - depends on the network
        from urllib.request import urlopen

        with urlopen(TINY_SHAKESPEARE_URL, timeout=timeout) as response:
            text = response.read().decode("utf-8")
    except Exception:
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    return text


def load_base_corpus(source: str = "tinytales") -> tuple[list[str], str]:
    """Return ``(documents, source_name)`` for the base corpus.

    ``source`` may be:

    ``"tinytales"``
        The bundled, generated storybook corpus.  No network, deterministic.
        This is the default because a live stream cannot afford a download.
    ``"shakespeare"``
        Public-domain tiny shakespeare, split into paragraphs.  Falls back to
        ``tinytales`` automatically if the download fails.
    """
    if source == "shakespeare":
        text = fetch_tiny_shakespeare()
        if text is not None:
            docs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 120]
            return docs, "shakespeare"
        # Fall through to the offline corpus.
    return tinytales_documents(), "tinytales"


# ---------------------------------------------------------------------------
# 2. Making it messy (so that cleaning it means something)
# ---------------------------------------------------------------------------

_BOILERPLATE = [
    "<div class=\"content-wrapper\"><p>read more</p></div>",
    "Copyright 2019 Example Media Group. ALL RIGHTS RESERVED.",
    "click here to subscribe to our newsletter!!!",
    "Share this on social media | Print | Email | Comments (0)",
    "<!-- BEGIN AD SLOT 3 -->",
    "cookie notice: this site uses cookies. accept | decline",
    "https://example.com/?utm_source=feed&utm_medium=rss",
]

_SHORT_JUNK = ["ok.", "hi", "...", "n/a", "TODO", "??", "first!"]

_MOJIBAKE_SWAPS = [
    ("'", "\u00e2\u0080\u0099"),
    (" - ", " \u00e2\u0080\u0093 "),
    ("e", "\u00c3\u00a9"),
    ("a", "\ufffd"),
]


def _mojibake(doc: str, rng: random.Random) -> str:
    """Simulate text that was decoded with the wrong character encoding."""
    src, dst = rng.choice(_MOJIBAKE_SWAPS)
    return "\u00ef\u00bb\u00bf" + doc.replace(src, dst, 6)


def _near_duplicate(doc: str, rng: random.Random) -> str:
    """A copy that differs only in whitespace / case / trailing punctuation."""
    out = doc
    if rng.random() < 0.5:
        out = out.upper() if rng.random() < 0.3 else out.capitalize()
    out = re.sub(r" ", "  " if rng.random() < 0.5 else " ", out)
    out = out.replace(".", ". " if rng.random() < 0.5 else ".")
    if rng.random() < 0.5:
        out = "  " + out + "   \n"
    if rng.random() < 0.4:
        out = out + " " + rng.choice(["thanks!", "(edited)", "[1]"])
    return out


def make_messy_corpus(clean_docs: Sequence[str], seed: int = 7) -> list[str]:
    """Return a shuffled, realistically dirty version of ``clean_docs``.

    Roughly a third of the result is garbage of one kind or another, which is
    not far off what a raw web scrape looks like.
    """
    rng = random.Random(seed)
    messy: list[str] = list(clean_docs)

    n = len(clean_docs)
    messy += [clean_docs[rng.randrange(n)] for _ in range(int(0.10 * n))]  # exact dupes
    messy += [_near_duplicate(clean_docs[rng.randrange(n)], rng) for _ in range(int(0.08 * n))]
    messy += [_mojibake(clean_docs[rng.randrange(n)], rng) for _ in range(int(0.05 * n))]
    messy += [rng.choice(_BOILERPLATE) for _ in range(int(0.05 * n))]
    messy += [rng.choice(_SHORT_JUNK) for _ in range(int(0.04 * n))]
    # A "wall of text" page: the same sentence pasted over and over.
    messy += [
        (clean_docs[rng.randrange(n)].split(".")[0] + ". ") * 60
        for _ in range(int(0.01 * n))
    ]
    rng.shuffle(messy)
    return messy


# ---------------------------------------------------------------------------
# 3. Cleaning, one explainable step at a time
# ---------------------------------------------------------------------------


@dataclass
class CleaningStep:
    """One row of the before/after cleaning table."""

    name: str
    why: str
    kept: int
    removed: int

    @property
    def removed_pct(self) -> float:
        total = self.kept + self.removed
        return 100.0 * self.removed / total if total else 0.0


def normalize_whitespace(docs: Iterable[str]) -> list[str]:
    """Collapse runs of spaces/newlines and trim the edges.

    Not a filter - it just makes every later step easier to reason about.
    """
    return [re.sub(r"\s+", " ", unicodedata.normalize("NFKC", d)).strip() for d in docs]


def broken_char_ratio(doc: str) -> float:
    """Fraction of characters that are not in our expected alphabet."""
    if not doc:
        return 1.0
    bad = sum(1 for ch in doc if ch not in ALLOWED_CHARS)
    return bad / len(doc)


def drop_broken_text(docs: Sequence[str], max_bad_ratio: float = 0.0) -> list[str]:
    """Throw away mojibake and other wrong-encoding wreckage.

    Our corpus is plain English, so *any* character outside the expected
    alphabet is a decoding accident, not content - hence the default of "zero
    tolerance".

    Why it matters: every weird character becomes a token the model must learn.
    A handful of garbage symbols can eat a noticeable slice of a tiny model's
    vocabulary and capacity for no benefit at all.
    """
    return [d for d in docs if broken_char_ratio(d) <= max_bad_ratio]


_BOILERPLATE_PATTERNS = re.compile(
    r"(<[^>]+>|https?://|utm_source|all rights reserved|copyright \d{4}"
    r"|click here|cookie notice|subscribe to our)",
    re.IGNORECASE,
)


def drop_boilerplate(docs: Sequence[str]) -> list[str]:
    """Remove navigation, ads, legal footers and raw HTML.

    Why it matters: boilerplate is the most-repeated text on the internet, so a
    model trained on it learns to write cookie banners instead of sentences.
    """
    return [d for d in docs if not _BOILERPLATE_PATTERNS.search(d)]


def filter_by_length(
    docs: Sequence[str], min_chars: int = 120, max_chars: int = 1200
) -> list[str]:
    """Keep documents that are long enough to teach and short enough to be real.

    Why it matters: two-word documents carry no context for a next-token model,
    and giant copy-pasted walls of text are almost always spam.
    """
    return [d for d in docs if min_chars <= len(d) <= max_chars]


def _fingerprint(doc: str) -> str:
    """Lower-cased, punctuation-free, whitespace-collapsed version of a doc."""
    return re.sub(r"[^a-z0-9 ]+", "", doc.lower())


def dedup_exact(docs: Sequence[str]) -> list[str]:
    """Remove byte-identical copies, keeping the first one seen.

    Why it matters: a document that appears 10 times is effectively trained on
    10 times harder.  That is how models end up reciting text verbatim.
    """
    seen: set[str] = set()
    out: list[str] = []
    for d in docs:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _shingles(doc: str, k: int = 4) -> set[str]:
    words = _fingerprint(doc).split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    """How much overlap two sets have: ``|A and B| / |A or B|``, from 0 to 1."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _minhash(shingles: set[str], n_hashes: int = 16) -> tuple[int, ...]:
    """A tiny, deterministic MinHash signature (no numpy, no randomness)."""
    sig = []
    for seed in range(n_hashes):
        prefix = f"{seed}:".encode()
        sig.append(min(zlib.crc32(prefix + s.encode("utf-8")) for s in shingles))
    return tuple(sig)


def dedup_near(
    docs: Sequence[str], threshold: float = 0.8, k: int = 4, bands: int = 4
) -> list[str]:
    """Remove documents that are *almost* copies of one we already kept.

    Uses MinHash + banding: cheap signatures group likely-similar documents into
    buckets, and we only run the exact Jaccard comparison inside a bucket.

    Why it matters: exact dedup misses the most common real-world case - the
    same article republished with a different headline, extra whitespace, or a
    "(edited)" tag stuck on the end.
    """
    kept: list[str] = []
    kept_shingles: list[set[str]] = []
    buckets: dict[tuple, list[int]] = {}
    n_hashes = 16
    rows = n_hashes // bands

    for doc in docs:
        sh = _shingles(doc, k=k)
        if not sh:
            continue
        sig = _minhash(sh, n_hashes)
        keys = [(b, sig[b * rows : (b + 1) * rows]) for b in range(bands)]
        candidates = {i for key in keys for i in buckets.get(key, ())}
        if any(jaccard(sh, kept_shingles[i]) >= threshold for i in candidates):
            continue
        index = len(kept)
        kept.append(doc)
        kept_shingles.append(sh)
        for key in keys:
            buckets.setdefault(key, []).append(index)
    return kept


def clean_corpus(
    raw_docs: Sequence[str],
    *,
    min_chars: int = 120,
    max_chars: int = 1200,
    near_dup_threshold: float = 0.8,
) -> tuple[list[str], list[CleaningStep]]:
    """Run the whole pipeline and return ``(clean_docs, per-step report)``."""
    steps: list[CleaningStep] = []
    docs = list(raw_docs)
    start = len(docs)

    stages = [
        ("normalise whitespace", "makes every later rule predictable", normalize_whitespace),
        ("drop broken characters", "mojibake wastes a tiny model's vocabulary", drop_broken_text),
        ("drop boilerplate", "ads and legal footers are not language", drop_boilerplate),
        ("length filter", "too short teaches nothing, too long is usually spam",
         lambda d: filter_by_length(d, min_chars, max_chars)),
        ("exact dedup", "repeats push the model toward memorising", dedup_exact),
        ("near dedup", "the same text with a tweak is still a repeat",
         lambda d: dedup_near(d, near_dup_threshold)),
    ]
    for name, why, fn in stages:
        before = len(docs)
        docs = list(fn(docs))
        steps.append(CleaningStep(name, why, kept=len(docs), removed=before - len(docs)))

    steps.append(
        CleaningStep("TOTAL", "raw -> clean", kept=len(docs), removed=start - len(docs))
    )
    return docs, steps


# ---------------------------------------------------------------------------
# 4. The train / validation split
# ---------------------------------------------------------------------------


def split_train_val(
    docs: Sequence[str], val_fraction: float = 0.1, seed: int = 0
) -> tuple[list[str], list[str]]:
    """Split documents into a training set and a held-out validation set.

    The **validation set** is the pop quiz: the model never trains on it, so its
    score there tells us whether the model actually learned the language or just
    memorised the homework.

    We split by *document*, not by character, so that no sentence can appear on
    both sides of the wall.
    """
    if not 0.0 < val_fraction < 0.5:
        raise ValueError("val_fraction should be a small positive fraction")
    shuffled = list(docs)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]


def docs_to_text(docs: Sequence[str]) -> str:
    """Join documents with a blank line so the model can learn where they end."""
    return "\n\n".join(docs) + "\n"


# ---------------------------------------------------------------------------
# 5. Instruction data for supervised fine-tuning (notebook 05)
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = "q: {question}\na: "

# Short on purpose: a whole example (prompt + answer) has to fit inside the
# model's 96-token context window.
_QA_TEMPLATES = [
    ("where did {name} find it?", "{name} found it near the {place}."),
    ("who followed {name} home?", "a {adj} {animal} followed {name} home."),
    ("what did {name} keep?", "{name} kept the {obj} in a pocket."),
    ("what is by the {place}?", "a {adj} {animal} is by the {place}."),
    ("how did {name} feel?", "{name} felt quite {adj} about it."),
    ("what can i cook with {veg}?", "make a {adj} {veg} {dish}."),
    ("when do i add the {herb}?", "add the {herb} right at the end."),
    ("how long does it keep?", "it keeps for {number} days in the fridge."),
    ("what goes in the pan first?", "heat the {fat} in a wide pan first."),
    ("how many does it serve?", "it serves {number} people."),
]


def qa_pairs(n_pairs: int = 900, seed: int = 99) -> list[dict[str, str]]:
    """Generated ``{"question": ..., "answer": ...}`` pairs for fine-tuning.

    Short, consistent and formulaic on purpose: the point of notebook 05 is that
    the model learns the *shape* of "someone asked, so I answer" - not new facts.
    """
    rng = random.Random(seed)
    pairs = []
    for _ in range(n_pairs):
        q_t, a_t = rng.choice(_QA_TEMPLATES)
        filled = {
            "name": rng.choice(_NAMES),
            "animal": rng.choice(_ANIMALS),
            "place": rng.choice(_PLACES),
            "obj": rng.choice(_OBJECTS),
            "adj": rng.choice(_ADJECTIVES),
            "veg": rng.choice(_VEG),
            "fat": rng.choice(_FAT),
            "herb": rng.choice(_HERB),
            "dish": rng.choice(_DISH),
            "number": rng.randint(2, 9),
        }
        pairs.append({"question": q_t.format(**filled), "answer": a_t.format(**filled)})
    return pairs


def format_example(question: str, answer: str) -> tuple[str, str]:
    """Return ``(prompt, completion)`` in the exact format the model will see.

    The completion ends with a newline: that is the model's "I am finished"
    signal, and it is why generation can stop cleanly after fine-tuning.
    """
    return PROMPT_TEMPLATE.format(question=question), answer + "\n"


# ---------------------------------------------------------------------------
# 6. Saving / loading the prepared splits
# ---------------------------------------------------------------------------


def save_splits(
    train_docs: Sequence[str],
    val_docs: Sequence[str],
    name: str = "base",
    directory: Path | None = None,
) -> dict[str, Path]:
    """Write ``<name>_train.txt`` / ``<name>_val.txt`` into ``data/``."""
    directory = directory or DATA
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": directory / f"{name}_train.txt",
        "val": directory / f"{name}_val.txt",
    }
    paths["train"].write_text(docs_to_text(train_docs), encoding="utf-8")
    paths["val"].write_text(docs_to_text(val_docs), encoding="utf-8")
    return paths


def load_split(name: str = "base", split: str = "train", directory: Path | None = None) -> str:
    directory = directory or DATA
    path = directory / f"{name}_{split}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing - run notebooks/01_data.ipynb (or scripts/prepare_data.py) first."
        )
    return path.read_text(encoding="utf-8")


def save_qa(pairs: Sequence[dict], name: str = "sft", directory: Path | None = None) -> Path:
    directory = directory or DATA
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.jsonl"
    path.write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in pairs) + "\n",
        encoding="utf-8",
    )
    return path


def load_qa(name: str = "sft", directory: Path | None = None) -> list[dict]:
    directory = directory or DATA
    path = directory / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing - run notebooks/01_data.ipynb first.")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# ---------------------------------------------------------------------------
# 7. One call that produces every file the later notebooks need
# ---------------------------------------------------------------------------

#: How many documents the deliberately-too-small "overfit me" training set gets.
TINY_TRAIN_DOCS = 6


def prepare_all(source: str = "tinytales", seed: int = 0, verbose: bool = True) -> dict:
    """Build and save every dataset the repository uses.

    Notebook ``01_data.ipynb`` walks through these same steps one at a time and
    explains them; this function exists so ``scripts/`` and the other notebooks
    can get a consistent dataset in one line.

    Returns a small summary dict (sizes, paths, vocabulary size).
    """
    from minigpt.tokenizer import CharTokenizer

    DATA.mkdir(parents=True, exist_ok=True)

    # --- base ("stories") domain -------------------------------------------
    base_docs, source_name = load_base_corpus(source)
    messy = make_messy_corpus(base_docs)
    clean, steps = clean_corpus(messy)
    base_train, base_val = split_train_val(clean, val_fraction=0.1, seed=seed)
    save_splits(base_train, base_val, "base")

    # --- new ("recipes") domain for continued pre-training ------------------
    rec = recipe_documents()
    rec_train, rec_val = split_train_val(rec, val_fraction=0.15, seed=seed)
    save_splits(rec_train, rec_val, "recipes")

    # --- the deliberately tiny training set used for the overfit demo -------
    tiny = base_train[:TINY_TRAIN_DOCS]
    (DATA / "tiny_train.txt").write_text(docs_to_text(tiny), encoding="utf-8")

    # --- instruction-tuning pairs ------------------------------------------
    pairs = qa_pairs()
    sft_train, sft_val = pairs[: int(0.9 * len(pairs))], pairs[int(0.9 * len(pairs)) :]
    save_qa(sft_train, "sft")
    save_qa(sft_val, "sft_val")

    # --- one shared vocabulary across every stage --------------------------
    sft_text = "".join(
        "".join(format_example(p["question"], p["answer"])) for p in pairs
    )
    everything = (
        docs_to_text(base_train) + docs_to_text(base_val)
        + docs_to_text(rec_train) + docs_to_text(rec_val)
        + sft_text
    )
    tokenizer = CharTokenizer.from_text(everything)
    tokenizer.save(DATA / "vocab.json")

    summary = {
        "source": source_name,
        "raw_docs": len(messy),
        "clean_docs": len(clean),
        "cleaning_steps": [s.__dict__ for s in steps],
        "base_train_docs": len(base_train),
        "base_val_docs": len(base_val),
        "base_train_chars": sum(len(d) for d in base_train),
        "base_val_chars": sum(len(d) for d in base_val),
        "recipes_train_docs": len(rec_train),
        "recipes_val_docs": len(rec_val),
        "tiny_train_docs": len(tiny),
        "tiny_train_chars": sum(len(d) for d in tiny),
        "sft_train_pairs": len(sft_train),
        "sft_val_pairs": len(sft_val),
        "vocab_size": tokenizer.vocab_size,
    }
    (DATA / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if verbose:
        print(f"corpus source        : {source_name}")
        print(f"raw -> clean docs    : {len(messy)} -> {len(clean)}")
        print(f"base train / val docs: {len(base_train)} / {len(base_val)}")
        print(f"recipes train / val  : {len(rec_train)} / {len(rec_val)}")
        print(f"sft pairs train / val: {len(sft_train)} / {len(sft_val)}")
        print(f"vocabulary size      : {tokenizer.vocab_size}")
        print(f"written to           : {DATA}")
    return summary


def load_tokenizer(directory: Path | None = None):
    """Load the shared vocabulary saved by :func:`prepare_all`."""
    from minigpt.tokenizer import CharTokenizer

    directory = directory or DATA
    path = directory / "vocab.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing - run notebooks/01_data.ipynb or "
            "`python scripts/prepare_data.py` first."
        )
    return CharTokenizer.load(path)


__all__ = [
    "ALLOWED_CHARS",
    "CleaningStep",
    "PROMPT_TEMPLATE",
    "TINY_TRAIN_DOCS",
    "broken_char_ratio",
    "clean_corpus",
    "dedup_exact",
    "dedup_near",
    "docs_to_text",
    "drop_boilerplate",
    "drop_broken_text",
    "fetch_tiny_shakespeare",
    "filter_by_length",
    "format_example",
    "jaccard",
    "load_base_corpus",
    "load_qa",
    "load_split",
    "load_tokenizer",
    "make_messy_corpus",
    "normalize_whitespace",
    "prepare_all",
    "qa_pairs",
    "recipe_documents",
    "save_qa",
    "save_splits",
    "split_train_val",
    "tinytales_documents",
]
