"""Nguyen-style "Simple Contrastive" probe: a difference-of-means direction scored by projection.

Nguyen et al. release no probe weights -- only a contrastive dataset and the generation
code. So the probe is *reconstructed*: direction = mean(test activations) - mean(deploy
activations) at a given layer, exactly as in their src/vector_generation.py. Scoring a
prompt = projecting its activation onto the unit probe direction (their src/probe_analysis.py
uses dot(act, v) / ||v||).

Activations are per-dimension standardized (z-scored) before projection. The probe direction
and threshold are built in the *contrastive set's* z-space; a held-out set is standardized
using stats fit on that held-out set itself. Fitting the standardizer per dataset is what
makes the threshold transfer: raw projections carry each set's common-mode offset along the
probe direction, which grows with depth and differs across datasets, so a train-derived
threshold lands off-scale on the eval set (the Phase 1 first-run bug: deep-layer thresholds
~2-4 vs 2x2 scores ~0). Centering each set on its own mean removes that offset, leaving the
threshold to measure relative eval-vs-deploy separation -- exactly the confound question.

Pure numpy -- no model dependency, so it is unit-testable locally.
"""
import numpy as np


def fit_standardizer(acts):
    """Per-dimension (mean, std) over (n, d) activations; zero-variance dims get std=1."""
    acts = np.asarray(acts, dtype=np.float64)
    mu = acts.mean(axis=0)
    sigma = acts.std(axis=0)
    sigma[sigma == 0] = 1.0
    return mu, sigma


def standardize(acts, mu, sigma):
    """Z-score activations with previously fit stats: (acts - mu) / sigma."""
    return (np.asarray(acts, dtype=np.float64) - mu) / sigma


def build_probe(test_acts, deploy_acts):
    """Raw probe direction = mean(test) - mean(deploy). acts: (n, d) arrays."""
    test_acts = np.asarray(test_acts, dtype=np.float64)
    deploy_acts = np.asarray(deploy_acts, dtype=np.float64)
    return test_acts.mean(axis=0) - deploy_acts.mean(axis=0)


def build_probe_weighted(acts, weights):
    """Soft-label diff-of-means: weighted_mean(acts, w) - weighted_mean(acts, 1-w).

    weights[i] in [0,1] = P(positive class) for activation i. The hard-label build_probe is
    the special case weights in {0,1}. Down-weights ambiguous (w~0.5) activations in both
    means, so hedged labels contribute little to the direction.
    """
    acts = np.asarray(acts, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    wp, wn = w.sum(), (1.0 - w).sum()
    pos = (acts * w[:, None]).sum(axis=0) / wp if wp > 0 else np.zeros(acts.shape[1])
    neg = (acts * (1.0 - w)[:, None]).sum(axis=0) / wn if wn > 0 else np.zeros(acts.shape[1])
    return pos - neg


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def project(acts, probe):
    """Project activations onto the unit probe direction (Nguyen: dot(act, v) / ||v||)."""
    return np.asarray(acts, dtype=np.float64) @ unit(probe)


def midpoint_threshold(test_acts, deploy_acts, probe):
    """Decision threshold = midpoint of the two contrastive class means projected onto the probe.

    This is the natural boundary for a difference-of-means classifier and needs no extra
    calibration set. (Nguyen calibrated their threshold on SAD stages_oversight via Youden's
    J; we don't reuse SAD here, and the midpoint is sign-stable by construction.)
    """
    t = project(test_acts, probe).mean()
    d = project(deploy_acts, probe).mean()
    return 0.5 * (t + d)


def classify_eval(scores, threshold):
    """Classify a prompt as 'eval/test' if its score sits on the contrastive test side.

    probe = mean(test) - mean(deploy), so test activations always project *higher* than
    deploy activations -- the test side is unambiguously the high side.
    """
    return np.asarray(scores) > threshold


def cosine(a, b):
    """Cosine similarity between two vectors."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def auroc(scores, labels):
    """Tie-averaged AUROC = P(score | label=1 > score | label=0). Pure numpy."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    avg_rank = (csum - counts + 1 + csum) / 2.0
    ranks = avg_rank[inv]
    return (ranks[labels == 1].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
