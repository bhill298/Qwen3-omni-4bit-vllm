@echo off
:: Start vLLM in a new minimized window that will kill container on close
start "vLLM Server" /MIN cmd /c "docker run --gpus all --name vllm-qwen-omni --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -it -v qwen_model_cache:/model -v vllm_compiler_cache:/root/.cache/vllm -v "C:\path\to\media:/media" -p 8000:8000 -e VLLM_SERVER_DEV_MODE=1 vllm-qwen3omni-patched:latest /model --trust-remote-code --tensor-parallel-size 1 --max-model-len 8192 --max-num-seqs 2 --disable-custom-all-reduce --no-enable-prefix-caching --enable-sleep-mode --allowed-local-media-path /media"

echo Waiting for vLLM to load...
:wait_loop
timeout /t 2 /nobreak >nul
curl -4 -s http://localhost:8000/health
if errorlevel 1 goto wait_loop

echo vLLM loaded. Putting to sleep...
curl -4 -X POST http://localhost:8000/sleep

echo vLLM sleeping. This window will close. Close the minimized "vLLM Server" window to stop the container.
timeout /t 3 >nul