"""Modal app: extract Llama-3.3-70B residual-stream activations for the refusal probe.

For a list of instructions, runs the target model and returns, per requested layer, the
**last-token resid_post** activation (the assistant-turn boundary), plus Arditi's refusal
score (log-odds of the first response token being "I") for refusal-score filtering.

Extraction (forward hook on decoder_layers[L].output[0]) is adapted from src/modal_app.py.
The probe direction itself is built off-worker with src/probe.py.

Entry point: scripts/part1_refusal_nla.py
"""
import modal

APP_NAME = "part1-refusal-extract"
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
GPU_CONFIG = "H100:2"
DEFAULT_LAYERS = [40, 53, 63]  # 53 = NLA layer (committed); 40/63 = health-check sweep
LLAMA3_REFUSAL_TOK = 40  # token id for "I" (Arditi LLAMA3_REFUSAL_TOKS)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers==4.49.0",
        "accelerate>=0.33",
        "huggingface_hub[hf_transfer]>=0.24",
        "numpy>=1.24",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("src")
)

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("nla-labelled-probes-hf-cache", create_if_missing=True)


@app.cls(
    gpu=GPU_CONFIG,
    image=image,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_dotenv()],
    timeout=2 * 60 * 60,
    scaledown_window=120,
)
class TargetModel:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left padding so the last position (index -1) is the true last token (Arditi).
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.model.eval()
        self.decoder_layers = self.model.model.layers
        self.input_device = self.model.get_input_embeddings().weight.device
        print(f"Loaded {MODEL_ID}: {len(self.decoder_layers)} decoder layers", flush=True)

    def _capture(self, input_ids, attention_mask, layers):
        """Forward pass; returns ({layer: resid_post (b,seq,d) cpu f32}, last-position logits)."""
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
                out = self.model(
                    input_ids=input_ids.to(self.input_device),
                    attention_mask=attention_mask.to(self.input_device),
                )
        finally:
            for h in handles:
                h.remove()
        return captured, out.logits[:, -1, :].detach().float().cpu()

    @modal.method()
    def extract(self, instructions, layers=None, batch_size=8):
        """Last-token resid_post per layer + refusal score per instruction.

        Returns {"acts": {str(L): (n,d) float16}, "refusal_scores": (n,) float32}.
        """
        import numpy as np
        import torch

        from src.data_refusal import format_prompt

        layers = layers or DEFAULT_LAYERS
        acts = {L: [] for L in layers}
        scores = []
        for i in range(0, len(instructions), batch_size):
            batch = [format_prompt(x) for x in instructions[i : i + batch_size]]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True)
            cap, last_logits = self._capture(enc.input_ids, enc.attention_mask, layers)
            for L in layers:
                hs = cap[L]  # (b, seq, d); left-padded so [-1] is the last real token
                acts[L].append(hs[:, -1, :].numpy())
            # refusal score = log p(I) - log(1 - p(I)) at last position (Arditi)
            probs = torch.softmax(last_logits, dim=-1)
            p_ref = probs[:, LLAMA3_REFUSAL_TOK]
            scores.append((torch.log(p_ref + 1e-8) - torch.log(1 - p_ref + 1e-8)).numpy())

        return {
            "acts": {str(L): np.concatenate(acts[L], axis=0).astype(np.float16) for L in layers},
            "refusal_scores": np.concatenate(scores, axis=0).astype(np.float32),
        }
