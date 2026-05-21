"""Refusal datasets + Llama-3 prompt formatting (Arditi et al. 2024).

Splits are vendored verbatim from github.com/andyrdt/refusal_direction (`dataset/splits/`):
harmful = AdvBench/TDC2023/HarmBench/MaliciousInstruct mix, harmless = Alpaca. We reuse his
exact files, chat template, and sampling so the refusal direction is a faithful reproduction.

Pure-python (json + random) so it runs locally and inside the Modal worker (no torch).
"""
import json
import os
import random

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "data", "refusal")

HARMTYPES = ("harmful", "harmless")
SPLITS = ("train", "val", "test")

# Arditi's Llama-3 chat template, no system prompt (his pipeline/model_utils/llama3_model.py).
# The tokenizer prepends <|begin_of_text|>. The text after {instruction} is the
# end-of-instruction scaffold; its final token is the assistant-turn boundary we extract at.
LLAMA3_CHAT_TEMPLATE = (
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)


def load_split(harmtype, split, instructions_only=True):
    """Load one vendored split -> list[str] (or list[{instruction, category}])."""
    assert harmtype in HARMTYPES and split in SPLITS
    with open(os.path.join(DATA_DIR, f"{harmtype}_{split}.json")) as f:
        data = json.load(f)
    return [d["instruction"] for d in data] if instructions_only else data


def format_prompt(instruction):
    """Wrap a raw instruction in Arditi's Llama-3 chat template."""
    return LLAMA3_CHAT_TEMPLATE.format(instruction=instruction)


def sample_train_val(n_train=128, n_val=32, seed=42):
    """Replicate Arditi's load_and_sample_datasets: seeded sample of train/val per class.

    Returns dict with harmful_train, harmless_train, harmful_val, harmless_val (lists of str).
    Refusal-score filtering happens later, on the worker, once activations/logits are available.
    """
    rng = random.Random(seed)
    return {
        "harmful_train": rng.sample(load_split("harmful", "train"), n_train),
        "harmless_train": rng.sample(load_split("harmless", "train"), n_train),
        "harmful_val": rng.sample(load_split("harmful", "val"), n_val),
        "harmless_val": rng.sample(load_split("harmless", "val"), n_val),
    }


def sample_heldout(n_each=100, seed=43):
    """Balanced held-out set from the test splits (distinct seed from train/val).

    Returns (instructions, labels) where label 1 = harmful, 0 = harmless.
    """
    rng = random.Random(seed)
    harmful = rng.sample(load_split("harmful", "test"), n_each)
    harmless = rng.sample(load_split("harmless", "test"), n_each)
    instructions = harmful + harmless
    labels = [1] * len(harmful) + [0] * len(harmless)
    return instructions, labels
