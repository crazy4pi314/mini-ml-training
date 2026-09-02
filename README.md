# mini-ml-training

A hands-on, beginner-friendly walkthrough of **pre-training, continued pre-training, and post-training a tiny language model** — start to finish, on a laptop CPU, in about an hour.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/crazy4pi314/mini-ml-training?quickstart=1)

No GPU. No large downloads. No maths on screen. Every piece of jargon is defined
in plain English the first time it shows up.

---

## What you'll actually build

A **165,000-parameter** character-level GPT that starts out producing noise and
ends up writing sentences and answering questions. Then you'll break it on
purpose — three times — so you can recognise each failure on a chart.

```
raw messy text
   |  clean it, dedupe it, split off a validation set        (01_data)
   v
train / validation splits
   |  next-token prediction, ~1200 steps                     (02_pretrain)
   v
BASE MODEL  ---------> underfit / healthy / overfit          (03_diagnosis)
   |  keep training on a new domain                          (04_continued_pretraining)
   v
DOMAIN MODEL
   |  supervised fine-tuning on question -> answer pairs     (05_posttraining)
   v
INSTRUCTION MODEL                                            (06_wrap_up)
```

---

## Quick start

### In a Codespace (recommended)

Click the badge above. The dev container installs CPU-only PyTorch and builds
the datasets automatically (~2 minutes), then open
`notebooks/00_setup_check.ipynb`.

### Locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/prepare_data.py     # builds data/          (~5 seconds)
jupyter lab                        # then open notebooks/00_setup_check.ipynb
```

Everything is already committed, including the pre-baked checkpoints, so you can
also just open notebook 03 and go.

---

## The notebooks, in order

| # | Notebook | What happens | Runtime |
|---|---|---|---|
| 00 | `00_setup_check.ipynb` | Versions, a CPU sanity check, a speed measurement, and building `data/` | **~10 s** |
| 01 | `01_data.ipynb` | Messy text -> clean corpus in six explainable steps; tokenization; the validation split | **~7 s** |
| 02 | `02_pretrain.ipynb` | Define and train the model live; samples at step 0 / half way / the end | **~1 min 50 s** |
| 03 | `03_diagnosis.ipynb` | **The important one.** Underfitting vs generalising vs overfitting, plus a memorisation test | **~9 s** pre-baked, ~6 min live |
| 04 | `04_continued_pretraining.ipynb` | New domain, catastrophic forgetting, and how to avoid it | **~9 s** pre-baked, ~2 min live |
| 05 | `05_posttraining.ipynb` | Supervised fine-tuning, loss masking, before/after answers, and where DPO fits | **~9 s** pre-baked, ~1 min live |
| 06 | `06_wrap_up.ipynb` | The one-page "I see this -> it means this -> I do this" reference | **~7 s** |

Measured end to end with `python scripts/run_notebooks.py` on a 4-thread CPU
(**2 min 38 s for all seven**), with `USE_PREBAKED = True` where offered.
Notebook 02 trains live by design — that's the bit people came to watch.

---

## The presenter's safety net

Notebooks 03, 04 and 05 each have a flag in an obvious cell near the top:

```python
USE_PREBAKED = True    # load results that were trained in advance
```

`scripts/pretrain_all.py` produces every checkpoint and every loss history ahead
of time and commits them to `checkpoints/` and `artifacts/`. With the flag on,
those notebooks run in seconds and you are never stuck watching a progress bar
on stream.

```bash
python scripts/pretrain_all.py             # all six runs, ~7 minutes
python scripts/pretrain_all.py --only healthy overfit
```

Everything is seeded (`seed=1337`), so re-running reproduces the same numbers and
the same generated text.

---

## Running the live stream

**Total: ~60 minutes.** Suggested pacing:

| Time | Segment | Notebook | The one thing to land |
|---|---|---|---|
| 0-3 | Welcome, what we're building | — | "By the end you'll have trained a language model on a CPU." |
| 3-5 | Environment check | 00 | No GPU needed. Ever. |
| 5-14 | **Data is the model** | 01 | Dedup and the validation split matter more than architecture. |
| 14-26 | **Train it live** | 02 | Watch gibberish -> words -> sentences. Talk over the progress log. |
| 26-40 | **Reading the curves** | 03 | Studying vs memorising the answer key. This is the segment to protect. |
| 40-48 | Continued pre-training | 04 | New skills can erase old ones — keep the old validation set. |
| 48-56 | Post-training | 05 | Autocomplete -> assistant. Loss stops being the scoreboard. |
| 56-60 | Wrap-up | 06 | The screenshot-able cheat sheet. |

**Before you go live:**

1. `python scripts/pretrain_all.py` — confirm `checkpoints/` and `artifacts/`
   are populated.
2. Run `00_setup_check.ipynb`; look at the reported ms/step. If 1200 steps would
   take more than ~3 minutes on the machine you're presenting from, set
   `USE_PREBAKED = True` in notebook 02 as well.
3. Restart all kernels so nothing carries over from rehearsal.
4. If notebook 02 runs long, cut the temperature-comparison cell near the end —
   nothing later depends on it.

**Presenter notes** are inline: markdown cells beginning with
`> **Presenter script.**` are written to be narrated more or less verbatim.

---

## Repository layout

```
minigpt/                  the shared module the notebooks import
  |- paths.py             where data/checkpoints/artifacts live
  |- tokenizer.py         CharTokenizer: text <-> token IDs
  |- data.py              corpus generation, mess injection, cleaning, splits, Q&A pairs
  |- model.py             MiniGPT - a ~165k-parameter nanoGPT-style transformer
  |- train.py             training loop, evaluation, sampling, SFT, checkpoints
  |- plots.py             the charts, styled for a stream

notebooks/                00 ... 06, one per live-stream segment
scripts/
  |- prepare_data.py      build data/ without opening a notebook
  |- pretrain_all.py      bake every checkpoint + loss history in advance
  |- run_notebooks.py     execute all notebooks headlessly (the test suite)
  |- clean_notebooks.py   strip outputs before committing

data/                     train/val splits, Q&A pairs, shared vocabulary  (~660 KB, committed)
checkpoints/              base, overfit, cpt_naive, cpt_replay, sft       (~3 MB, committed)
artifacts/                loss histories as JSON for the pre-baked path   (~15 KB, committed)
.devcontainer/            Codespaces definition (Python 3.11, CPU-only)
```

### Why are checkpoints committed?

They total about 3 MB — small enough that committing them is cheaper than making
every presenter (and every viewer following along) spend seven minutes baking
them. Regenerate at any time with `python scripts/pretrain_all.py`.

---

## About the dataset

The default corpus is **TinyTales**: ~1,400 short stories generated from a
seeded template grammar in `minigpt/data.py`.

That is a deliberate trade-off, and worth being upfront about:

* **No download, ever.** Identical on every machine, works offline, and cannot
  break the stream because a CDN is having a bad day.
* **It learns fast.** A 165k-parameter character model produces recognisable
  English on this corpus in ~2 minutes of CPU time. On real prose it would need
  far longer to get anywhere as legible — which would be a much worse demo.
* **The mess is real.** `make_messy_corpus()` injects genuine duplicates,
  near-duplicates, mojibake, boilerplate and junk, so the cleaning in notebook 01
  removes actual problems rather than staged ones.

If you'd rather use real prose, the public-domain *tiny shakespeare* corpus
(~1 MB) is one argument away and falls back to TinyTales automatically if the
download fails:

```bash
python scripts/prepare_data.py --source shakespeare
python scripts/pretrain_all.py --skip-data
```

Expect higher loss values and less polished samples for the same training budget.

---

## Verifying the repo

```bash
python scripts/run_notebooks.py        # executes all 7 notebooks, reports timings
```

Exit code 0 means every notebook ran top to bottom without raising.

---

## Jargon, all in one place

| Term | Plain English |
|---|---|
| **token** | The smallest chunk of text the model sees. Here: one character. |
| **vocabulary** | The full list of tokens the model knows (79 for us). |
| **parameter** | One adjustable number inside the model. We have ~165,000. |
| **loss** | How surprised the model was by the character that actually came next. Lower is better. |
| **step** | Look at one batch, make one small adjustment. |
| **batch size** | How many chunks of text per step. |
| **epoch** | One full pass over the training set. |
| **learning rate** | How big each adjustment is. |
| **context window** | How far back the model can see (96 characters). |
| **training set** | The homework. The model sees it over and over. |
| **validation set** | The pop quiz. The model never trains on it. |
| **underfitting** | It hasn't finished learning yet. |
| **overfitting** | It memorised the homework instead of learning the subject. |
| **checkpoint** | A saved copy of every parameter. |
| **dropout** | Randomly ignoring some internal values during training, to discourage memorising. |
| **pre-training** | Learning language itself by predicting the next token. |
| **continued pre-training** | Same thing, but on a new domain, starting from an existing checkpoint. |
| **catastrophic forgetting** | Learning something new wipes out something old. |
| **replay** | Mixing old data back in so the model doesn't forget it. |
| **post-training / SFT** | Teaching the model to follow instructions using prompt->response pairs. |
| **loss masking** | Grading the model on the answer only, never on the question. |
| **preference tuning (DPO/RLHF)** | Teaching it which of two good answers people prefer. |

---

## Where to go next

* [nanoGPT](https://github.com/karpathy/nanoGPT) — the direct big brother of this repo.
* [Let's build GPT (Karpathy)](https://www.youtube.com/watch?v=kCc8FmEb1nY) — builds this model line by line.
* [Hugging Face NLP course](https://huggingface.co/learn/nlp-course) — real tokenizers, real models.
* [`trl`](https://huggingface.co/docs/trl) — SFT and DPO implemented properly.
* [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) — notebook 01, at fifteen trillion tokens.

---

## License

MIT. The corpus is generated by this repository, so there's nothing to attribute.
