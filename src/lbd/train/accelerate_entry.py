from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def _is_true(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _log(message: str) -> None:
    print(f"[lbd.train] {message}", flush=True)


def _extract_resolution(argv: list[str]) -> int:
    for index, value in enumerate(argv):
        if value != "--resolution":
            continue
        if index + 1 >= len(argv):
            break
        try:
            return int(argv[index + 1])
        except ValueError:
            break
    return 512


def _disable_cudnn(reason: str) -> None:
    import torch

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    _log(f"cuDNN disabled: {reason}")


def _run_cuda_preflight(argv: list[str]) -> None:
    import torch

    if not torch.cuda.is_available():
        return

    if _is_true(os.environ.get("LBD_DISABLE_CUDNN")):
        _disable_cudnn("requested by LBD_DISABLE_CUDNN")
        return

    resolution = _extract_resolution(argv)
    probe_size = max(64, min(resolution, 512))
    try:
        with torch.no_grad():
            probe = torch.zeros((1, 3, probe_size, probe_size), device="cuda", dtype=torch.float16)
            conv = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1).cuda().half()
            _ = conv(probe)
        del conv
        del probe
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except RuntimeError as exc:
        if "CUDNN_STATUS" not in str(exc).upper():
            raise
        _disable_cudnn(f"preflight failed with: {exc}")
        torch.cuda.empty_cache()


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: accelerate_entry.py <target_script> [args...]")

    target_script = Path(sys.argv[1]).resolve()
    if not target_script.exists():
        raise FileNotFoundError(f"Target script not found: {target_script}")

    forwarded_argv = [str(target_script), *sys.argv[2:]]
    _run_cuda_preflight(forwarded_argv)
    sys.argv = forwarded_argv
    runpy.run_path(str(target_script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
