#!/usr/bin/env python3
"""
qwen3_omni_batch.py - Batch process images/videos/audio through Qwen3-Omni on vLLM.

This script expects the media files to ALREADY be located inside the
configured media directory, because it passes direct file:///media/... URIs to
the vLLM Docker container.
"""

import argparse
import json
import mimetypes
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

requests.packages.urllib3.util.connection.HAS_IPV6 = False


CONFIG = {
    "llama_server": os.environ.get("LLAMA_SERVER_URL", "http://localhost:8099"),
    "vllm_server": os.environ.get("VLLM_SERVER_URL", "http://localhost:8000"),
    "media_dir": os.environ.get("QWEN_MEDIA_DIR"),
    "model": "/model",
    "request_timeout": int(os.environ.get("QWEN_REQUEST_TIMEOUT", "300")),
    "max_tokens": int(os.environ.get("QWEN_MAX_TOKENS", "2048")),
    "temperature": float(os.environ.get("QWEN_TEMPERATURE", "0.7")),
}

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".m4v",
    ".wmv",
    ".flv",
    ".mpeg",
    ".mpg",
    ".ogv",
    ".ts",
    ".3gp",
}
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".webp",
    ".tiff",
    ".tif",
    ".ico",
    ".tga",
}
AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".opus",
    ".amr",
}


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Model lifecycle management
# ---------------------------------------------------------------------------


def unload_llama_models():
    """Unload all loaded models from the llama.cpp server to free VRAM."""
    log("=== Unloading llama.cpp models ===")
    try:
        resp = requests.get(f"{CONFIG['llama_server']}/v1/models", timeout=10)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        loaded = [m["id"] for m in models if m["status"]["value"] == "loaded"]
        if not loaded:
            log("No loaded models found on llama.cpp server.")
            return
        for model_id in loaded:
            log(f"Unloading: {model_id}")
            r = requests.post(
                f"{CONFIG['llama_server']}/models/unload",
                json={"model": model_id},
                timeout=30,
            )
            r.raise_for_status()
            log(f"  -> OK")
        # poll until all models are no longer loaded
        MAX_ATTEMPTS = 20
        for _ in range(MAX_ATTEMPTS):
            check = requests.get(
                f"{CONFIG['llama_server']}/v1/models",
                timeout=10,
            )
            check.raise_for_status()
            models = check.json().get("data", [])
            loaded = [m["id"] for m in models if m["status"]["value"] == "loaded"]
            if len(loaded) == 0:
                break
            time.sleep(0.5)
        else:
            raise RuntimeError(
                f"Llama models still loaded after checking for {MAX_ATTEMPTS} attempts"
            )
        # probably not needed
        time.sleep(0.1)
    except requests.exceptions.ConnectionError:
        log("Warning: could not connect to llama.cpp server. Continuing.")
    except Exception as e:
        log(f"Warning: couldn not unload llama models ({e}).")
        raise


def _server_alive():
    try:
        r = requests.get(f"{CONFIG['vllm_server']}/is_sleeping", timeout=5)
        if r.status_code == 200:
            return r.json().get("is_sleeping") is False
        return False
    except Exception:
        return False


def wake_vllm():
    """Wake up the vLLM/Qwen3-Omni server. Ok if already awake."""
    try:
        resp = requests.post(f"{CONFIG['vllm_server']}/wake_up", timeout=120)
        resp.raise_for_status()
        log("vLLM server woken up.")
        return True
    except Exception as e:
        log(
            f"Error waking vLLM server (server is most likely not running or wrong url): {e}."
        )
        return False


def sleep_vllm():
    """Put the vLLM server to sleep to free VRAM for the llama model."""
    log("=== Sleeping vLLM server ===")
    try:
        resp = requests.post(f"{CONFIG['vllm_server']}/sleep", timeout=60)
        resp.raise_for_status()
        log("vLLM server is now sleeping.")
    except Exception as e:
        log(f"Warning: could not sleep vLLM server ({e}). Continuing.")


# ---------------------------------------------------------------------------
# API call & Processing
# ---------------------------------------------------------------------------


def get_media_type(path):
    """Determine media type (video, audio, or image) using progressive fallbacks."""
    path = Path(path)
    suffix = path.suffix.lower()

    # Determine base type
    base_type = "image"
    mime, _ = mimetypes.guess_type(str(path))
    if mime:
        if mime.startswith("video/"):
            base_type = "video"
        elif mime.startswith("audio/"):
            base_type = "audio"
        elif mime.startswith("image/"):
            base_type = "image"
    else:
        if suffix in VIDEO_EXTENSIONS:
            base_type = "video"
        elif suffix in AUDIO_EXTENSIONS:
            base_type = "audio"
        elif suffix in IMAGE_EXTENSIONS:
            base_type = "image"

    if base_type == "audio":
        return "audio"

    # Use ffprobe to count packets and definitively classify
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-count_packets",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_packets,codec_type",
            "-of",
            "json",
            str(path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(res.stdout)
        streams = info.get("streams", [])

        if streams:
            v_stream = streams[0]
            nb_packets = v_stream.get("nb_read_packets")
            if nb_packets is not None:
                if int(nb_packets) > 1:
                    return "video"
                else:
                    return "image"

        # If no video stream, check for audio
        cmd_audio = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "json",
            str(path),
        ]
        res_audio = subprocess.run(
            cmd_audio, capture_output=True, text=True, check=True
        )
        if json.loads(res_audio.stdout).get("streams"):
            return "audio"

    except Exception:
        pass

    return base_type


def is_video(path):
    return get_media_type(path) == "video"


def is_audio(path):
    return get_media_type(path) == "audio"


def query_model(messages, timeout=None):
    """Send a chat completion request. Returns the response text."""
    timeout = timeout or CONFIG["request_timeout"]
    payload = {
        "model": CONFIG["model"],
        "messages": messages,
        "max_tokens": CONFIG["max_tokens"],
        "temperature": CONFIG["temperature"],
    }
    resp = requests.post(
        f"{CONFIG['vllm_server']}/v1/chat/completions",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def process_task(task, args):
    """Process one media+prompt task."""
    media_path = Path(task["media"]).resolve()
    media_dir = Path(CONFIG["media_dir"]).resolve()

    # ---------------------------------------------------------
    # Preprocessing: Scale/Truncate media in-place with ffmpeg
    # ---------------------------------------------------------
    max_img = task.get("max_image_size", args.max_image_size)
    max_vid = task.get("max_video_size", args.max_video_size)
    fps = task.get("video_fps", args.video_fps)
    max_dur = task.get("max_video_duration", args.max_video_duration)

    vid = is_video(media_path)
    aud = is_audio(media_path)
    tmp_path = None

    try:
        if vid:
            log(
                f"  Preprocessing video: max_size={max_vid}, fps={fps}, max_duration={max_dur}"
            )
            # Output to .mp4 to ensure libx264 compatibility (e.g. for .webm inputs)
            tmp_path = media_path.with_suffix(".tmp.mp4")

            cmd = ["ffmpeg", "-i", str(media_path), "-y"]
            if max_dur > 0:
                cmd.extend(["-t", str(max_dur)])

            vf = f"fps={fps},scale='min({max_vid},iw)':'min({max_vid},ih)':force_original_aspect_ratio=decrease,scale=trunc(iw/2)*2:trunc(ih/2)*2"
            cmd.extend(
                [
                    "-vf",
                    vf,
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "18",
                    str(tmp_path),
                ]
            )

            res = subprocess.run(cmd, capture_output=True, check=False)
            if res.returncode == 0:
                try:
                    media_path.unlink()
                except OSError:
                    pass
                media_path = media_path.with_suffix(".mp4")
                tmp_path.rename(media_path)
                task["media"] = str(media_path)  # update task with new extension
            else:
                log(
                    f"  Warning: Video preprocessing failed: {res.stderr.decode('utf-8', errors='ignore')}"
                )
                if tmp_path.exists():
                    tmp_path.unlink()

        elif aud:
            log(f"  Preprocessing audio: no downsampling required")

        else:
            log(f"  Preprocessing image: max_size={max_img}")
            tmp_path = media_path.with_name(f".tmp_{media_path.name}")
            cmd = ["ffmpeg", "-i", str(media_path), "-y"]
            vf = f"scale='min({max_img},iw)':'min({max_img},ih)':force_original_aspect_ratio=decrease"
            cmd.extend(["-vf", vf, str(tmp_path)])

            res = subprocess.run(cmd, capture_output=True, check=False)
            if res.returncode == 0:
                tmp_path.replace(media_path)
            else:
                log(
                    f"  Warning: Image preprocessing failed: {res.stderr.decode('utf-8', errors='ignore')}"
                )
                if tmp_path.exists():
                    tmp_path.unlink()

    except Exception as e:
        log(f"  Warning: Preprocessing threw an exception: {e}")
        try:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass

    # Enforce that the file is in the mounted media directory
    try:
        rel_media_path = media_path.relative_to(media_dir)
    except ValueError:
        raise ValueError(f"Media file must be copied to {media_dir} first.")

    prompt_file = Path(task["prompt"]).resolve()
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_text = f.read().strip()

    media_uri = f"file:///media/{rel_media_path.as_posix()}"
    log(f"  Processing: {media_path.name} -> {media_uri}")

    content: list[dict] = [{"type": "text", "text": prompt_text}]

    if vid:
        content.append({"type": "video_url", "video_url": {"url": media_uri}})

        # Extract audio next to the video if it exists
        audio_path = media_path.with_name(f"{media_path.stem}_audio.wav")
        log(f"  Extracting audio -> {audio_path.name}")
        res = subprocess.run(
            ["ffmpeg", "-i", str(media_path), "-y", str(audio_path)],
            capture_output=True,
            check=False,
        )
        if (
            res.returncode == 0
            and audio_path.exists()
            and audio_path.stat().st_size > 0
        ):
            audio_uri = f"file:///media/{audio_path.relative_to(media_dir).as_posix()}"
            content.append({"type": "audio_url", "audio_url": {"url": audio_uri}})
        else:
            log(
                f"  No valid audio stream found or extraction failed. Skipping audio track."
            )

    elif aud:
        content.append({"type": "audio_url", "audio_url": {"url": media_uri}})

    else:
        content.append({"type": "image_url", "image_url": {"url": media_uri}})

    # Cast to object type for requests json
    messages = [{"role": "user", "content": list(content)}]

    log(f"    Sending request to Qwen3-Omni...")
    response = query_model(messages)
    log(f"    Response received ({len(response)} chars)")

    return {
        "media": task["media"],
        "prompt": task["prompt"],
        "prompt_text": prompt_text,
        "response": response,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Qwen3-Omni Batch Media Processor")
    parser.add_argument(
        "tasks_file", nargs="?", help="JSON tasks file (reads stdin if omitted)"
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=768,
        help="Max longest edge for images (default 768)",
    )
    parser.add_argument(
        "--max-video-size",
        type=int,
        default=512,
        help="Max longest edge for video frames (default 512)",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=4.0,
        help="Frame rate for video downsampling (default 4.0)",
    )
    parser.add_argument(
        "--max-video-duration",
        type=float,
        default=-1.0,
        help="Max seconds of video to process, -1 for full video (default -1)",
    )
    parser.add_argument(
        "--skip-llama-unload",
        action="store_true",
        help="Skip unloading llama.cpp models (useful if running a non-local model)",
    )
    args = parser.parse_args()

    if args.tasks_file:
        with open(args.tasks_file, "r", encoding="utf-8") as f:
            tasks = json.load(f)
    else:
        tasks = json.load(sys.stdin)

    if not tasks:
        log("No tasks provided.")
        print(json.dumps([]))
        return

    log(f"Loaded {len(tasks)} task(s)")

    for task in tasks:
        if "media" not in task or "prompt" not in task:
            log(f"ERROR: each task needs 'media' and 'prompt' fields")
            sys.exit(1)
        if not os.path.isfile(task["prompt"]):
            log(f"ERROR: prompt file not found: {task['prompt']}")
            sys.exit(1)

    if CONFIG["media_dir"] is None:
        log(
            "ERROR: QWEN_MEDIA_DIR environment variable is not set. Set this to the media directory path."
        )
        sys.exit(1)

    os.makedirs(CONFIG["media_dir"], exist_ok=True)
    results = []

    try:
        if not args.skip_llama_unload:
            # this waits until the model shows unloaded
            unload_llama_models()
        # need to wake before we send any requests (or else they will block in the queue)
        # fine if it's already awake so no need to check
        if not wake_vllm():
            # bail, server is most likely not running
            os._exit(1)

        print("waiting for server to come back up...")
        # the wake up message is blocking, but this doesn't hurt to have just in case
        while not _server_alive():
            time.sleep(2)

        for i, task in enumerate(tasks):
            log(f"\n--- Task {i + 1}/{len(tasks)} ---")
            try:
                result = process_task(task, args)
                results.append(result)
            except Exception as e:
                log(f"  ERROR: {e}")
                results.append(
                    {
                        "media": task["media"],
                        "prompt": task["prompt"],
                        "error": str(e),
                    }
                )

    finally:
        if not args.skip_llama_unload:
            sleep_vllm()

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
