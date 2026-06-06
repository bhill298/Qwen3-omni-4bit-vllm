# Qwen3-Omni on vLLM: Setup & Optimization Guide

This guide covers setting up, optimizing, and operating the **cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit** model using vLLM on a Windows/WSL2 system with an NVIDIA RTX 5090 (Blackwell architecture).

## Understanding How vLLM Handles the Architecture

The model's `config.json` contains `"architectures": ["Qwen3OmniMoeForConditionalGeneration"]`, which typically implies audio+text support. However, vLLM has its own internal registry (`registry.py`) that maps this string to `Qwen3OmniMoeThinkerForConditionalGeneration` — the text-only thinker variant. This means vLLM currently ignores all `talker.` and `code_predictor.` weights; they are never loaded into VRAM.

If a future vLLM update adds audio output support and you want to guarantee text-only mode defensively, edit the `config.json` inside your model folder and change:
```json
  "architectures": ["Qwen3OmniMoeForConditionalGeneration"]
```
to:
```json
  "architectures": ["Qwen3OmniMoeThinkerForConditionalGeneration"]
```
Because vLLM maps both strings to the exact same Python class, this makes your intent explicit and protects you from future changes.

---

## Setting Up the Environment (Fresh Start)

### Build the Patched vLLM Image
Because the RTX 5090 (`sm_120`) is a cutting-edge Blackwell architecture, upstream vLLM currently has two bugs requiring patches (Marlin MoE atomics and a CUDA seqlen crash in the Vision wrappers).

1. Ensure the `apply_patch.py` and `Dockerfile` are in your directory.
2. Build the patched image:
   ```bash
   docker build -t vllm-qwen3omni-patched:latest .
   ```
*(Note: Once vLLM officially merges the fixes for `sm_120` Vision + Marlin atomics and adds audio dependencies for multimodal input, you can abandon this Dockerfile and simply use `vllm/vllm-openai:latest`.)*

### Clone the Model & Fix Config
1. Download the AWQ model:
   ```bash
   git clone https://huggingface.co/cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit
   ```
2. Open `config.json` inside the model folder and make the following fix to prevent a Pydantic crash (we already applied this, but good to know for a fresh setup):
   - Replace `"rope_type": "default"` with `"rope_type": "mrope"` inside the `talker_config` -> `text_config` -> `rope_scaling` section.
   - Delete the legacy `"type": "default"` line immediately below it to avoid conflicts.

### Create Docker Volumes for Model and Cache (Crucial for Speed)
To avoid the Windows cross-filesystem slowdown and save time on repeated compilations, we create two native Docker volumes.
```bash
# Create a persistent native docker volume for the model weights
docker volume create qwen_model_cache

# Create a persistent volume for the vLLM PyTorch/Triton compiler cache
docker volume create vllm_compiler_cache

# Copy the model from your Windows path into the native volume
# (This takes a moment, but you only do it once)
docker run --rm -v "C:\path\to\Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit:/source_model" -v qwen_model_cache:/model alpine cp -r /source_model/. /model/
```

---

## Running the Server

Launch the container using the native volumes and optimization flags. To enable dynamic unloading, we pass `-e VLLM_SERVER_DEV_MODE=1`.

```bash
# If using Git Bash on Windows, run this first to prevent path string corruption
export MSYS_NO_PATHCONV=1

# Start the server (replace C:\path\to\your\media with the one you actually want to use)
docker run --gpus all --name vllm-qwen-omni --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -it -v qwen_model_cache:/model -v vllm_compiler_cache:/root/.cache/vllm -v "C:\path\to\your\media:/media" -p 8000:8000 -e VLLM_SERVER_DEV_MODE=1 vllm-qwen3omni-patched:latest /model --trust-remote-code --tensor-parallel-size 1 --max-model-len 8192 --max-num-seqs 2 --disable-custom-all-reduce --no-enable-prefix-caching --enable-sleep-mode --allowed-local-media-path /media
```

### Tradeoffs
- `--max-model-len 8192`: Caps context size to save massive amounts of KV Cache VRAM. Increase to `16384` or more if you need to pass long videos or text docs, but watch your VRAM usage.
- `--max-num-seqs 2`: Drastically limits the scheduler's memory profiling overhead, and tricks vLLM into skipping 90% of its heavy CUDA graph captures during startup.
- `--disable-custom-all-reduce`: Avoids wasting time initializing multi-GPU IPC networks.
- `--no-enable-prefix-caching`: Disables tracking of memory block hashes. Because you send unique videos/images each time, the caching provides no benefit and tracking it slows startup.
- `-v qwen_model_cache:/model`: Uses the fast volume. If you want to test updates to the weights, you must recopy them or revert to the slow Windows mount.
- `-v vllm_compiler_cache:/root/.cache/vllm`: Saves the compiled PyTorch/Triton binaries so subsequent boots skip the 10-20s JIT compilation phase.

---

## How to Use the Model

### Base64 vs Direct File Paths
When sending media to the API, you have two choices:
* **Base64 (`data:image/jpeg;base64,...`)**: Use this ONLY for small images or if the client and server are on completely different physical machines across the internet. Sending video files this way causes massive memory spikes (a 150MB video creates a 200MB JSON string, eating up 600MB+ of RAM during parsing) and blocks the server.
* **Direct File Paths (`file:///media/...`)**: Use this for videos, audio, or when the client and server share the same filesystem via Docker mounts. It is infinitely faster and drastically reduces RAM spikes!

To use direct file paths, mount your media folder (e.g., `-v "C:\path\to\your\media:/media"`) and whitelist it (`--allowed-local-media-path /media`) in your docker run command, then pass the file URI as shown below.

### Standard Chat / Multimodal Inference (Text & Image)
You can query the model using standard OpenAI-compatible endpoints.

**Text only:**
```bash
curl -s -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "/model", "messages": [{"role": "user", "content": "Explain quantum computing in one sentence."}], "max_tokens": 50, "temperature": 0.7}'
```

**Image input (via direct file path):**
```python
import requests

payload = {
    "model": "/model",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {"type": "image_url", "image_url": {"url": "file:///media/image.jpeg"}}
            ]
        }
    ]
}
requests.post("http://localhost:8000/v1/chat/completions", json=payload)
```

**Image input (via base64):**
```python
import requests, base64

def encode_file(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

payload = {
    "model": "/model",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_file('image.jpeg')}"}}
            ]
        }
    ]
}
requests.post("http://localhost:8000/v1/chat/completions", json=payload)
```


### Advanced Multimodal Inference (Video + Audio)
Qwen3-Omni handles both video frames and audio natively. However, the OpenAI API specification requires you to separate the audio stream from the video file and pass them as two distinct objects in the `content` array (`video_url` and `input_audio`).

*First, extract the audio from your video (e.g., using ffmpeg):*
```bash
ffmpeg -i video.mp4 -y audio.wav
```

*Then, send both to the API using direct file paths:*
```python
import requests

payload = {
    "model": "/model",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this video in detail, including the audio/sounds."},
                # reads the files directly from the disk
                {"type": "video_url", "video_url": {"url": "file:///media/video.mp4"}},
                {"type": "input_audio", "input_audio": {"url": "file:///media/audio.wav"}}
            ]
        }
    ]
}
requests.post("http://localhost:8000/v1/chat/completions", json=payload)
```

### Dynamically Unloading/Loading to free VRAM

Because we set `VLLM_SERVER_DEV_MODE=1`, vLLM enables internal memory-management endpoints.

**1. Sleep the Model (Free VRAM):**
Tell vLLM to go to sleep. This offloads the active weights/allocations from the GPU directly to your system CPU RAM.
```bash
curl -X POST http://localhost:8000/sleep
```
*Your 32GB VRAM is now free.*

**2. Wake the Model (Restore VRAM):**
When your agent needs Qwen again, wake it up. Because pulling from CPU RAM is orders of magnitude faster than reading from SSD, it resumes near-instantly.
```bash
curl -X POST http://localhost:8000/wake_up
```

**3. Check Status:**
```bash
curl -X GET http://localhost:8000/is_sleeping
# Returns {"is_sleeping": true/false}
```

## Skill
I added a skill in the skills/ directory to use this vllm server. It is hard-coded to talk to llama.cpp and load/unload itself and the vllm model. Set these environment vars:
- `LLAMA_SERVER_URL`
- `VLLM_SERVER_URL`
- `QWEN_MEDIA_DIR`

## TODO
This warning slows down inference, tracked in this issue https://github.com/vllm-project/vllm/issues/43009
```
(EngineCore pid=239) WARNING 06-06 21:48:28 [jit_monitor.py:103] Triton kernel JIT compilation during inference: rotary_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.
```
