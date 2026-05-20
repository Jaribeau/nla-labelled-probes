"""Modal app for Phase 1: a faithful replication of Devbunova (2026) on Llama-3.3-70B.

Devbunova's method (her scripts/evaluate_probe_cached.py), reproduced here with only the
model and layers changed:
  - Train a sklearn LogisticRegression on bench_eval (class 0 = eval) vs casual_deploy
    (class 1 = deploy), with a StandardScaler fit on the training activations.
  - Features = the *last-token* residual stream at the layer's *output* (resid_post),
    extracted from raw prompts (no chat template, no system prompt).
  - Test on the four held-out 2x2 cells; report % predicted class 0 ("eval").

The whole pipeline runs inside one warm container (`run_phase1`) so the 70B model is
loaded exactly once. Raw last-token activations are written to a Modal Volume for reuse;
only the trained probe + per-cell scores are returned to the caller.

Entry point: scripts/reproduce_format_confound.py
"""
import modal

APP_NAME = "phase1-reproduce-format-confound"
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"

# Llama-3.3-70B in bf16 is ~141GB of weights. 4xH100 (320GB) is comfortable headroom for
# weights + batched activations; H100:2 (160GB) is tight but may work. Trade cost vs. safety.
GPU_CONFIG = "H100:2"

# Devbunova swept layers 8-31 on the 32-layer Llama-3.1-8B (resid_post, 0-based) and found
# the format effect was layer-robust. On the 80-layer 70B we sweep an analogous spread;
# 53 = the layer of Neuronpedia's Llama-3.3-70B NLA, needed for Phase 3 alignment.
DEFAULT_LAYERS = [23, 40, 53, 63]

OUT_DIR = "/out"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers==4.49.0",
        "accelerate>=0.33",
        "huggingface_hub[hf_transfer]>=0.24",
        "datasets>=2.20",
        "numpy>=1.24",
        "scikit-learn>=1.3",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("src")
)

app = modal.App(APP_NAME)

# Persists the ~141GB model download. Project-scoped (not phase-scoped) so later phases
# share the same HF cache.
hf_cache = modal.Volume.from_name("nla-labelled-probes-hf-cache", create_if_missing=True)
# Phase 1 outputs: raw last-token activations + a copy of the result JSON, kept for inspection.
out_vol = modal.Volume.from_name("phase1-format-confound-out", create_if_missing=True)


@app.cls(
    gpu=GPU_CONFIG,
    image=image,
    volumes={"/root/.cache/huggingface": hf_cache, OUT_DIR: out_vol},
    secrets=[modal.Secret.from_dotenv()],  # picks up HF_TOKEN from local .env
    timeout=2 * 60 * 60,
    scaledown_window=120,
)
class Llama:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Right padding so attention_mask.sum()-1 indexes the true last token (Devbunova).
        self.tokenizer.padding_side = "right"
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.model.eval()
        self.decoder_layers = self.model.model.layers
        self.input_device = self.model.get_input_embeddings().weight.device
        print(f"Loaded {MODEL_ID}: {len(self.decoder_layers)} decoder layers", flush=True)

    # --- activation extraction --------------------------------------------------

    def _capture_resid_post(self, input_ids, attention_mask, layers):
        """Forward pass capturing the residual stream *leaving* each requested decoder layer.

        Matches Devbunova: a forward hook on decoder_layers[L] taking output[0]. Returns
        {layer: hidden_states (b, seq, d)} on CPU as float32.
        """
        torch = self.torch
        captured = {}
        handles = []

        def make_hook(L):
            def hook(module, inputs, output):
                hs = output[0] if isinstance(output, tuple) else output
                captured[L] = hs.detach().float().cpu()

            return hook

        for L in layers:
            handles.append(self.decoder_layers[L].register_forward_hook(make_hook(L)))
        try:
            with torch.no_grad():
                self.model(
                    input_ids=input_ids.to(self.input_device),
                    attention_mask=attention_mask.to(self.input_device),
                )
        finally:
            for h in handles:
                h.remove()
        return captured

    def _last_token_acts(self, prompts, layers, batch_size):
        """Last-token resid_post activations for raw prompts. Returns {layer: (n, d) array}."""
        import numpy as np

        acts = {L: [] for L in layers}
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            # Raw prompts: add_special_tokens=True prepends BOS, as in Devbunova's tokenizer call.
            enc = self.tokenizer(batch, return_tensors="pt", padding=True)
            cap = self._capture_resid_post(enc.input_ids, enc.attention_mask, layers)
            for L in layers:
                hs = cap[L]  # (b, seq, d) float32 cpu
                for r in range(hs.shape[0]):
                    n = int(enc.attention_mask[r].sum())
                    acts[L].append(hs[r, n - 1, :].numpy())
        return {L: np.stack(acts[L]) for L in layers}

    # --- probe training (Devbunova) ---------------------------------------------

    def _train_probe(self, layers, batch_size):
        """Train Devbunova's LogisticRegression probe per layer on bench_eval vs casual_deploy."""
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        from src.data import DEVBUNOVA_TRAIN_DEPLOY, DEVBUNOVA_TRAIN_EVAL, load_devbunova_2x2

        cells = load_devbunova_2x2()
        eval_prompts = cells[DEVBUNOVA_TRAIN_EVAL]
        deploy_prompts = cells[DEVBUNOVA_TRAIN_DEPLOY]
        print(f"Training probe: {len(eval_prompts)} {DEVBUNOVA_TRAIN_EVAL} (class 0) vs "
              f"{len(deploy_prompts)} {DEVBUNOVA_TRAIN_DEPLOY} (class 1)", flush=True)

        eval_acts = self._last_token_acts(eval_prompts, layers, batch_size)
        deploy_acts = self._last_token_acts(deploy_prompts, layers, batch_size)

        probes = {}
        for L in layers:
            X = np.concatenate([eval_acts[L], deploy_acts[L]], axis=0).astype(np.float32)
            y = np.array([0] * len(eval_acts[L]) + [1] * len(deploy_acts[L]))
            scaler = StandardScaler().fit(X)
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(scaler.transform(X), y)
            train_acc = float(clf.score(scaler.transform(X), y))
            probes[L] = {"clf": clf, "scaler": scaler, "train_acc": train_acc}
            print(f"  layer {L}: train_acc={train_acc:.3f}", flush=True)
        return probes

    # --- 2x2 evaluation ---------------------------------------------------------

    def _evaluate(self, probes, layers, batch_size, run_id):
        """Apply the trained probe to the four held-out 2x2 test cells."""
        import numpy as np
        from src.data import DEVBUNOVA_TEST_CELLS, load_devbunova_2x2

        cells = load_devbunova_2x2()
        results = {}
        pooled_npz = {}
        for cell_name in DEVBUNOVA_TEST_CELLS:
            prompts = cells[cell_name]
            print(f"Evaluating cell {cell_name}: {len(prompts)} prompts", flush=True)
            acts = self._last_token_acts(prompts, layers, batch_size)

            cell_res = {}
            for L in layers:
                clf = probes[L]["clf"]
                scaler = probes[L]["scaler"]
                X = scaler.transform(acts[L].astype(np.float32))
                pred = clf.predict(X)
                prob_deploy = clf.predict_proba(X)[:, 1]  # P(class 1 = deploy)
                cell_res[L] = {
                    "n": int(len(pred)),
                    "pct_eval": float((pred == 0).mean()),
                    "pct_deploy": float((pred == 1).mean()),
                    "mean_prob_deploy": float(prob_deploy.mean()),
                    "std_prob_deploy": float(prob_deploy.std()),
                }
                pooled_npz[f"{cell_name}__L{L}__last"] = acts[L].astype(np.float16)
            results[cell_name] = cell_res

        # Run-scoped filename so a later run never clobbers this run's activations.
        npz_path = f"{OUT_DIR}/phase1_pooled_activations_{run_id}.npz"
        np.savez_compressed(npz_path, **pooled_npz)
        out_vol.commit()
        print(f"Saved last-token activations to {npz_path}", flush=True)
        return results

    @modal.method()
    def run_phase1(self, layers=None, batch_size=8, run_id=None):
        """Full Phase 1 pipeline in one warm container. Returns a compact result dict."""
        import datetime
        import json

        layers = layers or DEFAULT_LAYERS
        run_id = run_id or datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        probes = self._train_probe(layers, batch_size)
        cell_results = self._evaluate(probes, layers, batch_size, run_id)

        result = {
            "model": MODEL_ID,
            "run_id": run_id,
            "layers": layers,
            "method": "devbunova-logreg (bench_eval vs casual_deploy, last-token resid_post, raw)",
            "probe": {
                str(L): {
                    "coef": probes[L]["clf"].coef_.tolist(),
                    "intercept": probes[L]["clf"].intercept_.tolist(),
                    "scaler_mean": probes[L]["scaler"].mean_.tolist(),
                    "scaler_scale": probes[L]["scaler"].scale_.tolist(),
                    "train_acc": probes[L]["train_acc"],
                }
                for L in layers
            },
            "cells": {c: {str(L): v for L, v in lv.items()} for c, lv in cell_results.items()},
        }
        # Persist a run-scoped full copy alongside the activations on the volume.
        with open(f"{OUT_DIR}/phase1_result_{run_id}.json", "w") as f:
            json.dump(result, f)
        out_vol.commit()
        return result
