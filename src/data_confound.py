"""Part 3 — planted MCQ-format confound on the refusal substrate.

Renders Arditi instructions in two formats with identical content: a multiple-choice scaffold
(the structural signature Devbunova found probes key on) vs free-form prose (Part 1's format).
The confounded training set makes format ≡ label (harmful-MCQ vs harmless-free); the
de-confounded 2×2 ({harmful,harmless} × {MCQ,free}) then exposes whether the probe tracks the
concept (harm) or the confound (MCQ format).

Pure-python; reuses src/data_refusal.load_split. The MCQ wrapper keeps the instruction as the
stem and adds the Question:/Choices:/(A)/(B)/Answer: scaffold — content unchanged, only format.
"""
import random

from src.data_refusal import load_split

MCQ_TEMPLATE = "Question: {instruction}\n\nChoices:\n(A) Yes\n(B) No\n\nAnswer:"


def mcq_wrap(instruction):
    """Wrap an instruction in a multiple-choice scaffold (format confound; content unchanged)."""
    return MCQ_TEMPLATE.format(instruction=instruction)


def free_wrap(instruction):
    """Free-form: the instruction as-is (Part 1 / Arditi format)."""
    return instruction


def build_confound_train(n_per_class=128, seed=42):
    """Confounded training set: harmful→MCQ (label 1), harmless→free (label 0). Format ≡ label.

    Returns {"pos": [mcq harmful...], "neg": [free harmless...]}.
    """
    rng = random.Random(seed)
    harmful = rng.sample(load_split("harmful", "train"), n_per_class)
    harmless = rng.sample(load_split("harmless", "train"), n_per_class)
    return {"pos": [mcq_wrap(x) for x in harmful], "neg": [free_wrap(x) for x in harmless]}


def build_2x2(n_per_cell=50, seed=43):
    """De-confounded 2×2: each held-out instruction rendered both ways. Distinct seed from train.

    Returns dict cell_name -> {"texts": [...], "instructions": [...], "harm": int, "mcq": int};
    harmful and harmless use the SAME instruction in both formats (content held fixed across
    the format axis), so the MCQ-vs-free contrast isolates format.
    """
    rng = random.Random(seed)
    harmful = rng.sample(load_split("harmful", "test"), n_per_cell)
    harmless = rng.sample(load_split("harmless", "test"), n_per_cell)
    cells = {}
    for harm, insts in ((1, harmful), (0, harmless)):
        for mcq, wrap in ((1, mcq_wrap), (0, free_wrap)):
            name = f"{'harmful' if harm else 'harmless'}_{'mcq' if mcq else 'free'}"
            cells[name] = {
                "texts": [wrap(x) for x in insts],
                "instructions": list(insts),
                "harm": harm,
                "mcq": mcq,
            }
    return cells
