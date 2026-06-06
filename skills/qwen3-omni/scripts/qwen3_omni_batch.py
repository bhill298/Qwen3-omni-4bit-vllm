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
        resp = requests.get(f"{CONFIG['llama_server']}/models", timeout=10)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        loaded = [m["id"] for m in models if m.get("status") == "loaded"]
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
    except requests.exceptions.ConnectionError:
        log("Warning: Could not connect to llama.cpp server. Continuing.")
    except Exception as e:
        log(f"Warning: Error unloading llama models ({e}). Continuing.")


def _server_alive():
    try:
        r = requests.get(f"{CONFIG['vllm_server']}/is_sleeping", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def wake_vllm():
    """Wake up the vLLM/Qwen3-Omni server. Ok if already awake."""
    log("=== Waking up vLLM server ===")
    if not _server_alive():
        log("vLLM server is not reachable. Will retry...")
        for attempt in range(5):
            time.sleep(2)
            if _server_alive():
                break
        else:
            log("Warning: vLLM server still not reachable. Continuing anyway.")

    try:
        r = requests.get(f"{CONFIG['vllm_server']}/is_sleeping", timeout=10)
        if r.status_code == 200 and r.json().get("is_sleeping") is False:
            log("vLLM server is already awake.")
            return
    except Exception:
        pass

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
        content.append({"type": "input_audio", "input_audio": {"url": audio_uri}})
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
            "ERROR: QWEN_MEDIA_DIR environment variable is not set. The LLM must set this to the media directory path."
        )
        sys.exit(1)

    os.makedirs(CONFIG["media_dir"], exist_ok=True)
    results = []

    try:
        unload_llama_models()
        wake_vllm()
        time.sleep(1)

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
