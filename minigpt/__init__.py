"""minigpt - the shared toolbox for the `mini-ml-training` live stream.

The notebooks stay short and readable because all of the plumbing lives here:

* :mod:`minigpt.paths`     - where things live on disk
* :mod:`minigpt.data`      - fetching, messing up, and cleaning the corpus
* :mod:`minigpt.tokenizer` - turning text into numbers (and back)
* :mod:`minigpt.model`     - a very small GPT
* :mod:`minigpt.train`     - the training loop, evaluation and text generation
* :mod:`minigpt.plots`     - the charts we read the training runs from
"""

from minigpt import data, model, paths, plots, tokenizer, train
from minigpt.model import DEFAULT_BATCH_SIZE, MiniGPT, MiniGPTConfig, default_config
from minigpt.tokenizer import CharTokenizer
from minigpt.train import (
    History,
    evaluate,
    generate,
    load_checkpoint,
    load_history,
    save_checkpoint,
    save_history,
    set_seed,
    train_model,
)

__all__ = [
    "CharTokenizer",
    "DEFAULT_BATCH_SIZE",
    "History",
    "MiniGPT",
    "MiniGPTConfig",
    "data",
    "default_config",
    "evaluate",
    "generate",
    "load_checkpoint",
    "load_history",
    "model",
    "paths",
    "plots",
    "save_checkpoint",
    "save_history",
    "set_seed",
    "tokenizer",
    "train",
    "train_model",
]

__version__ = "1.0.0"
