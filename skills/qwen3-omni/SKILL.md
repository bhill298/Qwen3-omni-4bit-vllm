---
name: qwen3-omni
description: Batch-process images and videos through the Qwen3-Omni multimodal model on vLLM to get text descriptions. Use when you need to describe images, analyze video content, or transcribe/describe video audio.
---

# Qwen3-Omni Multimodal Processing

This skill lets you temporarily offload VRAM from your local llama.cpp model to the Qwen3-Omni multimodal model (running via vLLM) in order to analyze images and videos.

## Agent Workflow

When you need to describe an image or video, you MUST follow these exact steps:

### 1. Copy Media Files
The vLLM server only has access to the directory specified by the `$QWEN_MEDIA_DIR` environment variable. **You must copy** the target image and video files into `$QWEN_MEDIA_DIR` before processing.

### 2. Create Prompt Files
Write the text prompts you want to use as `.txt` files (e.g., `$QWEN_MEDIA_DIR/prompt.txt`). Example: `"Describe this video in detail, paying attention to the audio."`

### 3. Create Tasks JSON
Create a `tasks.json` array containing the paths to the media and prompts you just placed in the media directory.

**Batching Guidelines & Time Estimates:**
- Batch multiple tasks together. Each script invocation results in the model getting loaded/unloaded, which adds overhead to multiple separate calls.
- Don't batch too many things together, otherwise interruptions will result in losing everything. Try to limit total execution time to an expected 1 hour max, if possible.
- **Videos:** Take roughly 30x their length to process (longer for higher resolution).
- **Images:** Take around 1 minute each (depending on size).

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
Run the bundled script located in the `scripts/` folder of this skill. If you are in the project root, the path is `.opencode/skills/qwen3-omni/scripts/qwen3_omni_batch.py`:

**IMPORTANT:** The script can take a long time. When running this script via the bash tool, set a very long timeout of at least 24 hours (e.g., `timeout: 86400000`) to ensure work isn't lost from the script being interrupted.

```bash
# You can also pass optional CLI arguments to set global defaults for the batch:
# --max-image-size 768 (default, max length of the longest edge in pixels)
# --max-video-size 512 (default, max length of the longest edge in pixels)
# --video-fps 2.0 (default)
# --max-video-duration -1.0 (default is -1, meaning process the whole video)
python .opencode/skills/qwen3-omni/scripts/qwen3_omni_batch.py tasks.json --max-video-duration 120.0
```

The script will automatically:
- Unload your llama.cpp model from VRAM
- Wake up vLLM
- Pass the files using `file:///media/...` direct transfer (and extract audio for videos automatically)
- Print a JSON array of the results to stdout
- Put vLLM back to sleep to return VRAM to you

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
python .opencode/skills/qwen3-omni/scripts/qwen3_omni_batch.py tasks.json

# 5. Clean up
rm "$QWEN_MEDIA_DIR/hero.mp4" "$QWEN_MEDIA_DIR/prompt.txt" "$QWEN_MEDIA_DIR/hero_audio.wav" tasks.json
```
