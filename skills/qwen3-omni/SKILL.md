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
   - **How to use it in JSON**: Backslashes will break JSON syntax. When generating `tasks.json` via bash, you MUST convert backslashes to forward slashes first using bash string replacement: `SAFE_DIR="${QWEN_MEDIA_DIR//\\//}"`, and use `$SAFE_DIR` in your JSON generation. Alternatively, write a small Python script to generate the JSON.
2. **Copy Media**: The vLLM server only accesses `$QWEN_MEDIA_DIR`. Copy target files here first.
3. **Create Prompts**: Write text prompts as `.txt` files in `$QWEN_MEDIA_DIR`.
4. **Create tasks.json**: JSON array mapping media to prompt files.

### 2. Pre-computation (Optional)
Check media details to estimate processing times and override default sizes if needed:
```bash
for f in "$QWEN_MEDIA_DIR"/*; do echo "$f:"; ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,duration -of default=noprint_wrappers=1:nokey=1 "$f"; done
```
*Output: Width, Height, Framerate, Duration (in seconds).* (Images show dummy values for duration and framerate).

### 3. Batching & Time Estimates
- Batch tasks to minimize model load/unload overhead. Limit total expected execution to ~1 hour max if possible.
- **Videos**: Takes ~3x to 5x video length (at 512px/2fps defaults). E.g., 5s video = ~15s processing. Processing raw, high-resolution/high-FPS video *without* these constraints can easily take upwards of 30x the video length and will eat up a lot of tokens.
- **Images**: ~10-15s each (at 768px default). High-res images without this limit can take over 1 minute each.
- **Audio**: Very fast, no downsampling.

```json
[
  {
    "media": "$SAFE_DIR/sunset.jpg",
    "prompt": "$SAFE_DIR/prompt.txt",
    "max_image_size": 768, "max_video_size": 512, "video_fps": 2.0, "max_video_duration": 120.0
  }
]
```
*(Overrides are optional. Sizes constrain the longest edge;aspect ratio is maintained).*

### 4. Execute Script
Run `scripts/qwen3_omni_batch.py` relative to the skill directory. **IMPORTANT**: Use a long bash timeout (e.g., `timeout: 86400000`) since the script can take a long time. Requires `requests` and `ffmpeg`.

```bash
# Optional CLI args: --max-image-size 768, --max-video-size 512, --video-fps 2.0, --max-video-duration -1.0, --skip-llama-unload
python scripts/qwen3_omni_batch.py tasks.json
```
*The script automatically manages model VRAM (unloads llama.cpp/wakes vLLM), preprocesses media in-place, queries the model, prints JSON results to stdout, and sleeps vLLM. --skip-llama-unload will skip the unload from llama and sleep vllm steps.*

### 5. Clean Up (CRITICAL)
Use bash to delete your copied media, prompt files, and generated `_audio.wav` files from `$QWEN_MEDIA_DIR` to prevent filling the disk.

## Example Usage
```bash
cp example.mp4 "$QWEN_MEDIA_DIR/example.mp4"
echo "What happens in this video?" > "$QWEN_MEDIA_DIR/prompt.txt"
SAFE_DIR="${QWEN_MEDIA_DIR//\\//}"
cat << EOF > tasks.json
[{"media": "$SAFE_DIR/example.mp4", "prompt": "$SAFE_DIR/prompt.txt"}]
EOF
python scripts/qwen3_omni_batch.py tasks.json
rm "$QWEN_MEDIA_DIR/example.mp4" "$QWEN_MEDIA_DIR/prompt.txt" "$QWEN_MEDIA_DIR/example_audio.wav" tasks.json
```
