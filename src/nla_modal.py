"""Modal app: serve the NLA activation verbalizer (AV) and verbalize activation vectors.

Hosts `kitft/Llama-3.3-70B-NLA-L53-av` via an in-container SGLang server (no public
Neuronpedia API), then uses the vendored `NLAClient` (src/nla.py) to inject each L53
activation into the AV's prompt embeddings and read back the `<explanation>`.

Runs separately from src/target_model.py — the two 70B models are never co-resident.

Entry point: scripts/part1_refusal_nla.py
"""
import modal

APP_NAME = "part1-nla-verbalize"
AV_REPO = "kitft/Llama-3.3-70B-NLA-L53-av"
GPU_CONFIG = "H100:4"  # 70B AV; tensor-parallel across all 4
N_GPU = 4
SGLANG_PORT = 30000

# Official sglang image: CUDA toolkit + torch + a prebuilt deep_gemm, all
# version-matched. Needed because sglang 0.5.6 unconditionally imports deep_gemm
# on Hopper (sm_90), which JIT-inits against a CUDA toolkit — absent from
# debian_slim, so the server crashed at startup (_find_cuda_home AssertionError).
# torch/transformers/numpy/safetensors ship in the base; only add the rest.
image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.6.post2")
    .pip_install(
        "httpx>=0.27",
        "orjson>=3.10",
        "pyyaml>=6.0",
        "huggingface_hub[hf_transfer]>=0.24",
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
class NLAVerbalizer:
    @modal.enter()
    def start(self):
        import subprocess
        import time

        import httpx
        from huggingface_hub import snapshot_download

        ckpt_path = snapshot_download(AV_REPO)
        print(f"Downloaded {AV_REPO} -> {ckpt_path}", flush=True)

        # SGLang server. --disable-radix-cache is REQUIRED for input_embeds (radix
        # cache keys on token ids, which embeds requests lack -> silent aliasing).
        self.proc = subprocess.Popen(
            [
                "python", "-m", "sglang.launch_server",
                "--model-path", ckpt_path,
                "--port", str(SGLANG_PORT),
                "--tp", str(N_GPU),
                "--disable-radix-cache",
                "--trust-remote-code",
                "--context-length", "2048",
                "--mem-fraction-static", "0.85",
            ]
        )

        url = f"http://localhost:{SGLANG_PORT}"
        deadline = time.time() + 20 * 60
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"SGLang exited early (code {self.proc.returncode})")
            try:
                if httpx.get(f"{url}/health_generate", timeout=5).status_code == 200:
                    break
            except Exception:
                time.sleep(5)
        else:
            raise RuntimeError("SGLang server did not become ready in time")
        print("SGLang ready", flush=True)

        from src.nla import NLAClient

        self.client = NLAClient(ckpt_path, sglang_url=url)

    @modal.method()
    def verbalize(self, vectors, temperature=0.0, max_new_tokens=256):
        """vectors: list of (d_model,) arrays -> list of explanation strings.

        temperature=0 = greedy/reproducible (the checkpoint's worked example uses temp=0).
        """
        import numpy as np

        vecs = [np.asarray(v, dtype=np.float32) for v in vectors]
        return self.client.generate_batch(
            vecs, temperature=temperature, max_new_tokens=max_new_tokens
        )

    @modal.exit()
    def stop(self):
        if getattr(self, "proc", None) is not None:
            self.proc.terminate()
