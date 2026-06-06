import re
import os

# 1. Force atomic add
path = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/marlin_utils.py"

with open(path, "r") as f:
    content = f.read()

old_func = """def should_use_atomic_add_reduce(
    m: int, n: int, k: int, device: torch.device, dtype: torch.dtype
) -> bool:
    # the performance of atomicAdd is better than global reduce
    # only when m*n is small and k is large
    if n >= 2048 or k < 2048 or device.type != "cuda":
        return False

    # disable atomicAdd reduce by default,
    # one can enable it with VLLM_MARLIN_USE_ATOMIC_ADD=1
    if not envs.VLLM_MARLIN_USE_ATOMIC_ADD:
        maybe_warn_marlin_atomic_add_env()
        return False

    # sm8x doesn't support atomicAdd + bfloat16 natively
    device_capability = torch.cuda.get_device_capability(device)
    if device_capability[0] < 9 and dtype == torch.bfloat16:
        maybe_warn_marlin_atomic_add(device, dtype)
        return False

    return True"""

new_func = """def should_use_atomic_add_reduce(
    m: int, n: int, k: int, device: torch.device, dtype: torch.dtype
) -> bool:
    return True"""

if old_func in content:
    content = content.replace(old_func, new_func)
    with open(path, "w") as f:
        f.write(content)
    print("Patched should_use_atomic_add_reduce in marlin_utils.py")

# 2. Patch flash_attn wrappers for cu_seqlens_q bug
attn_path = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/vit_attn_wrappers.py"
if os.path.exists(attn_path):
    with open(attn_path, "r") as f:
        attn_content = f.read()
    
    attn_content = attn_content.replace(
        "cu_seqlens_q=cu_seqlens,",
        "cu_seqlens_q=cu_seqlens.to('cuda') if cu_seqlens is not None else None,"
    )
    attn_content = attn_content.replace(
        "cu_seqlens_k=cu_seqlens,",
        "cu_seqlens_k=cu_seqlens.to('cuda') if cu_seqlens is not None else None,"
    )
    
    with open(attn_path, "w") as f:
        f.write(attn_content)
    print("Patched vit_attn_wrappers.py")

