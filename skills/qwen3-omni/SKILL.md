---
name: qwen3-omni
description: Batch-process images, audio, and videos through the Qwen3-Omni multimodal model on vLLM to get text descriptions. Use when you need to describe images, analyze video content, or transcribe/describe audio.
---

# Qwen3-Omni Multimodal Processing

Lets you do media analysis with Qwen3-omni (via vLLM) and automatically offloads VRAM to swap between local models.

## Agent Workflow

### 1. Setup Files
1. **Check Environment**: Ensure the `$QWEN_MEDIA_DIR` environment variable is set. This is a required variable. If it is empty or undefined, you must stop and ask the user to provide the directory path.
   - **CRITICAL for Windows/MSYS Bash**: The variable likely contains Windows paths with backslashes and spaces (e.g., `C:\Users\Name\Media`).
   - **How to check and verify**: Run `printenv QWEN_MEDIA_DIR` to see the value. Verify the directory exists by running `ls -ld "$QWEN_MEDIA_DIR"`.
   - **How to use it in Bash**: ALWAYS enclose the variable in double quotes (e.g., `cp media.mp4 "$QWEN_MEDIA_DIR/"`) in all bash commands. Do not attempt to manually escape backslashes, just use double quotes.
2. **Copy Media**: The vLLM server only accesses `$QWEN_MEDIA_DIR`. Copy target files here first.
3. **Create Prompts**: Write text prompts as `.txt` files in `$QWEN_MEDIA_DIR`.
4. **Create tasks.json**: JSON array mapping media to prompt files. Use the provided helper script `scripts/generate_tasks.py` to avoid JSON escaping issues with Windows backslashes.

### 2. Pre-computation (Optional)
Check media details to estimate processing times and override default sizes if needed:
```bash
for f in "$QWEN_MEDIA_DIR"/*; do echo "$f:"; ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,duration -of default=noprint_wrappers=1:nokey=1 "$f"; done
```
*Output: Width, Height, Framerate, Duration (in seconds).* (Images show dummy values for duration and framerate).

### 3. Batching & Time Estimates
- Batch tasks to minimize model load/unload overhead. Limit total expected execution to ~1 hour max if possible.
- **Videos**: Takes ~0.3x video length (at 512px/4fps defaults). E.g., 60s video = ~18s processing. Processing raw, high-resolution/high-FPS video *without* these constraints can take much longer and will eat up a lot of tokens.
  - **Striding for Long Videos**: You can use `stride_interval` and `stride_length` to capture intervals across long videos (e.g. 1 second every 60 seconds) without blowing up the context window.
- **Images**: ~3s each (at 768px default). High-res images without this limit will take much longer.
- **Audio**: Fast, no downsampling.

```json
[
  {
    "media": "C:/Users/Name/Media/sunset.jpg",
    "prompt": "C:/Users/Name/Media/prompt.txt",
    "max_image_size": 768, "max_video_size": 512, "video_fps": 4.0, "max_video_duration": 120.0,
    "stride_interval": -1.0, "stride_length": 1
  }
]
```
*(Overrides are optional. Sizes constrain the longest edge;aspect ratio is maintained).*

### 4. Execute Script
Run `scripts/qwen3_omni_batch.py` relative to the skill directory. Requires `requests` and `ffmpeg`.

**CRITICAL TIMEOUT INSTRUCTION**: If a tool call supports a timeout, you MUST provide a value of 2 hours (7200 seconds or 7200000 ms). For this script, always set the timeout to 2 hours. Do not use the default timeout.

Example tool call (if the tool call takes a timeout in seconds):
```json
{
  "command": "python scripts/qwen3_omni_batch.py tasks.json",
  "description": "Run Qwen3-Omni batch script",
  "timeout": 7200
}
```

```bash
# Optional CLI args: --max-image-size 768, --max-video-size 512, --video-fps 4.0, --max-video-duration -1.0, --stride-interval -1.0, --stride-length 1, --skip-llama-unload
python scripts/qwen3_omni_batch.py tasks.json
```
*The script automatically manages model VRAM (unloads llama.cpp/wakes vLLM), preprocesses media in-place, queries the model, prints JSON results to stdout, and sleeps vLLM. --skip-llama-unload will skip the unload from llama and sleep vllm steps.*

### 5. Clean Up (CRITICAL)
Use bash to delete your copied media, prompt files, and generated `_audio.wav` files from `$QWEN_MEDIA_DIR` to prevent filling the disk.

## Example Usage
```bash
cp example.mp4 "$QWEN_MEDIA_DIR/example.mp4"
echo "What happens in this video?" > "$QWEN_MEDIA_DIR/prompt.txt"

# Generate tasks.json using the helper script to avoid escaping issues
python scripts/generate_tasks.py "$QWEN_MEDIA_DIR/example.mp4" "$QWEN_MEDIA_DIR/prompt.txt" > tasks.json

# Execute the batch script (REMEMBER to set timeout: 2 hours in your tool call!)
python scripts/qwen3_omni_batch.py tasks.json

rm "$QWEN_MEDIA_DIR/example.mp4" "$QWEN_MEDIA_DIR/prompt.txt" "$QWEN_MEDIA_DIR/example_audio.wav" tasks.json
```
