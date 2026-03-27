"""Deploy vLLM on Modal for EasyInference demo testing.

Usage:
    modal deploy demo/modal_vllm.py

This deploys Qwen2.5-7B-Instruct on a single A10G GPU with:
- OpenAI-compatible /v1/chat/completions endpoint
- Prometheus /metrics endpoint for InferScope profiling
- Prefix caching enabled

The endpoint URL will be printed after deployment.
Change MODEL_ID and gpu= to test different configurations.
"""

import modal

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

app = modal.App("easyinference-demo")

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm")
)


@app.function(
    image=vllm_image,
    gpu="A10G",
    timeout=3600,
)
@modal.concurrent(max_inputs=100)
@modal.web_server(port=8000, startup_timeout=600)
def serve():
    import subprocess

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_ID,
        "--port", "8000",
        "--gpu-memory-utilization", "0.90",
        "--enable-prefix-caching",
        "--max-model-len", "4096",
        "--dtype", "auto",
    ]
    subprocess.Popen(cmd)
