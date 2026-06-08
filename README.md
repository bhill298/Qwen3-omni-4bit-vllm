# Qwen3-Omni on vLLM: Setup & Optimization Guide

This guide covers setting up, optimizing, and operating the **cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit** model using vLLM on a Windows/WSL2 system with an NVIDIA RTX 5090 (Blackwell architecture).

## Setting Up the Environment (Fresh Start)

### Build the Patched vLLM Image
The RTX 5090 (`sm_120`) Blackwell architecture has two upstream vLLM bugs requiring patches (Marlin MoE atomics and a CUDA seqlen crash in the Vision wrappers). This fixes those in the meantime. Additionally, this installs some missing audio handling dependencies for python.

Build the patched image:
```bash
docker build -t vllm-qwen3omni-patched:latest .
```
*(Note: Once vLLM merges the fixes for `sm_120` Vision + Marlin atomics and adds audio dependencies for multimodal input, you can directly use `vllm/vllm-openai:latest`.)*

### Clone the Model & Fix Config
1. Download the AWQ model:
```bash
git clone https://huggingface.co/cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit
```
2. Open `config.json` inside the model folder and make the following fix to prevent a Pydantic crash:
   - Replace `"rope_type": "default"` with `"rope_type": "mrope"` inside the `talker_config` -> `text_config` -> `rope_scaling` section.
   - Delete the legacy `"type": "default"` line immediately below it to avoid conflicts.

### Create Docker Volumes for Model and Cache
To avoid the Windows cross-filesystem slowdown and save time on repeated compilations, create two Docker volumes.
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

Launch the container using the docker volumes and optimization flags. To enable dynamic unloading, we pass `-e VLLM_SERVER_DEV_MODE=1` and `--enable-sleep-mode`. Alternatively, you can modify the provided `run_qwen_omni_vllm.template.bat` file on a Windows host, which by default launches then immediately puts vllm to sleep.

```bash
# If using Git Bash on Windows, run this first to prevent path string corruption
export MSYS_NO_PATHCONV=1

# Start the server (set %QWEN_MEDIA_DIR% to the host dir you want to use to store temp files)
docker run --gpus all --name vllm-qwen-omni --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -it -v qwen_model_cache:/model -v vllm_compiler_cache:/root/.cache/vllm -v "%QWEN_MEDIA_DIR%:/media" -p 8000:8000 -e VLLM_SERVER_DEV_MODE=1 vllm-qwen3omni-patched:latest /model --trust-remote-code --tensor-parallel-size 1 --max-model-len 16384 --max-num-seqs 2 --disable-custom-all-reduce --no-enable-prefix-caching --enable-sleep-mode --allowed-local-media-path /media
```

### Performance Considerations
- `--max-model-len 16384`: Caps context size to save massive amounts of KV Cache VRAM. Can consider increasing if you need to pass long videos or text docs.
- `--max-num-seqs 2`: Reduces CUDA graph capture size at startup and reduces KV cache memory overhead, in exchange for less batch parallelism.
- `--disable-custom-all-reduce`: Avoids wasting time initializing multi-GPU IPC networks.
- `--no-enable-prefix-caching`: Disables tracking of memory block hashes. If you send unique videos/images each time, the caching provides no benefit and tracking it slows startup.
- `-v qwen_model_cache:/model`: Uses the fast docker volume. If you want to test updates to the weights, you must recopy them or revert to the slower Windows mount.
- `-v vllm_compiler_cache:/root/.cache/vllm`: Saves the compiled PyTorch/Triton binaries so subsequent boots skip the 10-20s JIT compilation phase.

---

## How to Use the Model

### Base64 vs Direct File Paths
When sending media to the API, you have two choices:
* **Direct File Paths (`file:///media/...`)**: Use this for videos, audio, or when the client and server share the same filesystem via Docker mounts.
* **Base64 (`data:image/jpeg;base64,...`)**: Use this for small images or if the client and server are on completely different physical machines across the internet. Sending video files this way may cause memory spikes (a 150MB video creates a 200MB JSON string, eating up 600MB+ of RAM during parsing) and could block the server.

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
                {"type": "audio_url", "audio_url": {"url": "file:///media/audio.wav"}}
            ]
        }
    ]
}
requests.post("http://localhost:8000/v1/chat/completions", json=payload)
```

### Dynamically Unloading/Loading to free VRAM

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
There is a skill in the skills/ directory to use this vllm server. It is hard-coded to talk to llama.cpp and load/unload itself and the vllm model. Set these environment vars:
- `QWEN_MEDIA_DIR`
- `LLAMA_SERVER_URL`
- `VLLM_SERVER_URL`

The skill also includes a script, which you can also run directly. Running the script requires python with requests installed and ffmpeg.

---

## Understanding How vLLM Handles the Architecture

The model's `config.json` contains `"architectures": ["Qwen3OmniMoeForConditionalGeneration"]`, which typically implies audio+text support. However, vLLM has its own internal registry (`registry.py`) that maps this string to `Qwen3OmniMoeThinkerForConditionalGeneration` — the text-only thinker variant. This means vLLM currently ignores all `talker.` and `code_predictor.` weights; they are never loaded into VRAM.

If a future change makes it so that this matters and you want to force the thinker variant to save memory, you can make this change to the model `config.json`:
```json
  "architectures": ["Qwen3OmniMoeForConditionalGeneration"]
```
to:
```json
  "architectures": ["Qwen3OmniMoeThinkerForConditionalGeneration"]
```

---

## TODO
This warning slows down inference, tracked in this issue https://github.com/vllm-project/vllm/issues/43009
```
(EngineCore pid=239) WARNING 06-06 21:48:28 [jit_monitor.py:103] Triton kernel JIT compilation during inference: rotary_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.
```
