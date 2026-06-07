#!/usr/bin/env python3
"""
qwen3_omni_batch.py - Batch process images/videos through Qwen3-Omni on vLLM.

This script expects the media files to ALREADY be located inside the
configured media directory, because it passes direct file:///media/... URIs to
the vLLM Docker container.
"""

import json
import sys
import os
import subprocess
import argparse
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

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv", ".flv", ".mpeg", ".mpg"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}


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
        loaded = [m["id"] for m in models if m['status']['value'] == "loaded"]
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
            loaded = [m["id"] for m in models if m['status']['value'] == "loaded"]
            if len(loaded) == 0:
                break
            time.sleep(0.5)
        else:
            raise RuntimeError(f"Llama models still loaded after checking for {MAX_ATTEMPTS} attempts")
        # probably not needed
        time.sleep(.1)
    except requests.exceptions.ConnectionError:
        log("Error could not connect to llama.cpp server. Continuing.")
    except Exception as e:
        log(f"Error unloading llama models ({e}).")
        raise


def _server_alive():
    try:
        r = requests.get(f"{CONFIG['vllm_server']}/is_sleeping", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def wake_vllm():
    """Wake up the vLLM/Qwen3-Omni server. Ok if already awake."""
    try:
        resp = requests.post(f"{CONFIG['vllm_server']}/wake_up", timeout=120)
        resp.raise_for_status()
        log("vLLM server woken up.")
    except Exception as e:
        log(f"Warning: Error waking vLLM server ({e}). Continuing.")


def sleep_vllm():
    """Put the vLLM server to sleep to free VRAM for the llama model."""
    log("=== Sleeping vLLM server ===")
    try:
        resp = requests.post(f"{CONFIG['vllm_server']}/sleep", timeout=60)
        resp.raise_for_status()
        log("vLLM server is now sleeping.")
    except Exception as e:
        log(f"Warning: Error sleeping vLLM server ({e}). Continuing.")


# ---------------------------------------------------------------------------
# API call & Processing
# ---------------------------------------------------------------------------

def is_video(path):
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


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


def process_task(task):
    """Process one media+prompt task."""
    media_path = Path(task["media"]).resolve()
    media_dir = Path(CONFIG["media_dir"]).resolve()
    
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

    vid = is_video(media_path)
    content: list[dict] = [{"type": "text", "text": prompt_text}]

    if vid:
        content.append({"type": "video_url", "video_url": {"url": media_uri}})
        
        # Extract audio next to the video
        audio_path = media_path.with_name(f"{media_path.stem}_audio.wav")
        log(f"  Extracting audio -> {audio_path.name}")
        subprocess.run(
            ["ffmpeg", "-i", str(media_path), "-y", str(audio_path)],
            capture_output=True,
            check=True,
        )
        audio_uri = f"file:///media/{audio_path.relative_to(media_dir).as_posix()}"
        content.append({"type": "audio_url", "audio_url": {"url": audio_uri}})
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
    parser.add_argument("tasks_file", nargs="?", help="JSON tasks file (reads stdin if omitted)")
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
        # this waits until the model shows unloaded
        unload_llama_models()
        # need to wake before we send any requests (or else they will block in the queue)
        # fine if it's already awake so no need to check
        wake_vllm()

        print('waiting for server to come back up...')
        # the wake up message is blocking, but this doesn't hurt to have just in case
        while not _server_alive():
            time.sleep(2)

        for i, task in enumerate(tasks):
            log(f"\n--- Task {i+1}/{len(tasks)} ---")
            try:
                result = process_task(task)
                results.append(result)
            except Exception as e:
                log(f"  ERROR: {e}")
                results.append({
                    "media": task["media"],
                    "prompt": task["prompt"],
                    "error": str(e),
                })

    finally:
        sleep_vllm()

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
