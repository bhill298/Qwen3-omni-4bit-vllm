---
name: qwen3-omni
description: Batch-process images, audio, and videos through the Qwen3-Omni multimodal model on vLLM to get text descriptions. Use when you need to describe images, analyze video content, or transcribe/describe audio.
---

# Qwen3-Omni Multimodal Processing

This skill lets you temporarily offload VRAM from your local llama.cpp model to the Qwen3-Omni multimodal model (running via vLLM) in order to analyze images, audio files, and videos.

## Agent Workflow

When you need to describe an image, audio, or video file, you MUST follow these exact steps:

### 1. Copy Media Files
The vLLM server only has access to the directory specified by the `$QWEN_MEDIA_DIR` environment variable. **You must copy** the target image, audio, and video files into `$QWEN_MEDIA_DIR` before processing.

### 2. Create Prompt Files
Write the text prompts you want to use as `.txt` files (e.g., `$QWEN_MEDIA_DIR/prompt.txt`). Example: `"Describe this video in detail, paying attention to the audio."`

### 3. Create Tasks JSON
Create a `tasks.json` array containing the paths to the media and prompts you just placed in the media directory.

### Pre-computation Checks
To accurately estimate batch processing times and ensure inputs won't produce too many tokens, you can use `ffprobe` (included with `ffmpeg`) to determine the resolution, frame rate, and duration of the media.

You can check multiple files at once using a bash loop to minimize tool calls:
```bash
for f in "$QWEN_MEDIA_DIR"/*; do
  echo "$f:";
  ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,duration -of default=noprint_wrappers=1:nokey=1 "$f"; 
  echo "---";
done
```
*Output order for each file: Width, Height, Framerate (e.g. 30/1), Duration (in seconds).*
*Note: Images will also report a dummy duration and framerate (usually 25/1 and 0.04).*

Use this information in conjunction with the batching guidelines below to decide if you need to override the `--max-image-size`, `--max-video-size`, or `--max-video-duration` arguments for the script.

**Batching Guidelines & Time Estimates:**
- Batch multiple tasks together. Each script invocation results in the model getting loaded/unloaded, which adds overhead to multiple separate calls.
- Don't batch too many things together, otherwise interruptions will result in losing everything. Try to limit total execution time to an expected 1 hour max, if possible.
- **Videos:** Take roughly 3x to 5x their length to process when using the default 512px/2fps downsampling constraints (e.g. a 5-second video takes ~15 seconds). Note that processing raw, high-resolution/high-FPS video *without* these constraints can easily take upwards of 30x the video length or cause OOM crashes.
- **Images:** Take around 10 to 15 seconds each when using the default 768px constraint. High-res images without this limit can take over 1 minute each.
- **Audio:** Audio files (e.g., .wav, .mp3) are processed very quickly and do not undergo downsampling.

```json
[
  {
    "media": "$QWEN_MEDIA_DIR/sunset.jpg",
    "prompt": "$QWEN_MEDIA_DIR/prompt.txt",
    "max_image_size": 768,
    "max_video_size": 512,
    "video_fps": 2.0,
    "max_video_duration": 120.0
  }
]
```
*(Note: `max_image_size`, `max_video_size`, `video_fps`, and `max_video_duration` are optional fields that override the script's global defaults for the batch. The "max" sizes denote the maximum length in pixels of the longest edge of the image or video; aspect ratio is strictly maintained.)*

### 4. Execute the Script
Run the bundled script located at `scripts/qwen3_omni_batch.py` relative to the skill directory:

**IMPORTANT:** The script can take a long time. When running this script via the bash tool, set a very long timeout of at least 24 hours (e.g., `timeout: 86400000`) to ensure work isn't lost from the script being interrupted.

```bash
# You can also pass optional CLI arguments to set global defaults for the batch:
# --max-image-size 768 (default, max length of the longest edge in pixels)
# --max-video-size 512 (default, max length of the longest edge in pixels)
# --video-fps 2.0 (default)
# --max-video-duration -1.0 (default is -1, meaning process the whole video)
# --skip-llama-unload (optional, skips unloading llama models and skip putting vllm to sleep if you are using a non-local model)
python scripts/qwen3_omni_batch.py tasks.json --max-video-duration 120.0
```

The script requires the python requests library and ffmpeg.

The script will automatically:
- Unload your llama.cpp model from VRAM (unless `--skip-llama-unload` is passed)
- Wake up vLLM
- Process and downsample the media files in-place using ffmpeg based on your batch parameters
- Pass the files using `file:///media/...` direct transfer (and extract audio for videos automatically)
- Print a JSON array of the results to stdout
- Put vLLM back to sleep to return VRAM to you (unless `--skip-llama-unload` is passed)

### 5. Clean Up (CRITICAL)
Once you have the results, **you must clean up the files you copied** to `$QWEN_MEDIA_DIR`. Use bash commands to delete your copied media files, prompt files, and any generated `_audio.wav` files from that directory to prevent it from filling up.

## Example Usage
```bash
# 1. Copy the target file
cp src/assets/hero.mp4 "$QWEN_MEDIA_DIR/hero.mp4"

# 2. Make the prompt
echo "What happens in this video?" > "$QWEN_MEDIA_DIR/prompt.txt"

# 3. Create the batch config
cat << EOF > tasks.json
[{"media": "$QWEN_MEDIA_DIR/hero.mp4", "prompt": "$QWEN_MEDIA_DIR/prompt.txt"}]
EOF

# 4. Run the script
python scripts/qwen3_omni_batch.py tasks.json

# 5. Clean up
rm "$QWEN_MEDIA_DIR/hero.mp4" "$QWEN_MEDIA_DIR/prompt.txt" "$QWEN_MEDIA_DIR/hero_audio.wav" tasks.json
```
