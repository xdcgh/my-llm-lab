#!/usr/bin/env python3
"""PyTorch 矩阵乘法基准：类型支持、前向、可选反向、显式转换路径。

在项目根目录运行：
    uv run python lessons/00_bench_mps.py
    uv run python lessons/00_bench_mps.py --backward
    uv run python lessons/00_bench_mps.py --conversions

结果是当前软件栈的有效吞吐量，不是硬件峰值或原生指令支持的证明。
仅依赖 torch 和 Python 标准库；不联网，不修改模型，不启动后台进程。
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

# 必须在 import torch 之前设置。禁止未实现的 MPS 算子自动回退 CPU。
# 这不限制 GPU 内核内部的精度转换，也不阻止下文明确标注的 CPU 转换实验。
_PREVIOUS_FALLBACK = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

try:
    import torch
except ImportError:
    raise SystemExit("缺少 PyTorch：请在项目目录执行 uv add torch，再运行本脚本。")

DTYPES = {
    "fp32": "float32",
    "fp16": "float16",
    "bf16": "bfloat16",
    "fp8_e4m3fn": "float8_e4m3fn",
    "fp8_e5m2": "float8_e5m2",
    "fp64": "float64",
}
# 宽松的教学用相对误差门槛，不是模型质量标准。
TOLERANCES = {"fp32": 1e-4, "fp16": 1e-2, "bf16": 5e-2,
              "fp8_e4m3fn": 0.3, "fp8_e5m2": 0.5, "fp64": 1e-10}
Shape = tuple[int, int, int]  # M, K, N: A[M,K] @ B[K,N] -> C[M,N]


def parse_shape(text: str) -> Shape:
    try:
        dims = tuple(int(v) for v in text.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("形状应类似 128x2048x2048，即 MxKxN。") from exc
    if len(dims) != 3 or min(dims) <= 0:
        raise argparse.ArgumentTypeError("形状必须包含三个正整数 MxKxN。")
    return dims  # type: ignore[return-value]


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", choices=["mps", "cpu"], default="mps")
    p.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 2048], help="方阵边长")
    p.add_argument("--shapes", type=parse_shape, nargs="+", help="指定 MxKxN 后取代 --sizes")
    p.add_argument("--dtypes", choices=list(DTYPES), nargs="+", default=list(DTYPES))
    p.add_argument("--backward", action="store_true", help="另外测前向+两侧梯度；不是完整训练")
    p.add_argument("--conversions", action="store_true", help="另外测显式类型转换端到端耗时")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--target-ms", type=float, default=150.0, help="每组计时目标，可因迭代上限而达不到")
    p.add_argument("--max-iters", type=int, default=32, help="每组最大迭代次数")
    p.add_argument("--max-estimated-gib", type=float, default=2.0, help="单案例估计内存预算；不是硬上限")
    p.add_argument("--output", type=Path, help="JSON 路径；已有文件不会覆盖")
    a = p.parse_args()
    if min(a.sizes) <= 0 or min(a.warmup, a.repeats, a.max_iters) < 1:
        p.error("尺寸、预热次数、重复次数和迭代上限必须为正整数。")
    if not math.isfinite(a.target_ms) or a.target_ms <= 0:
        p.error("--target-ms 必须是正有限数。")
    if not math.isfinite(a.max_estimated_gib) or a.max_estimated_gib <= 0:
        p.error("--max-estimated-gib 必须是正有限数。")
    return a


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def cleanup(device: str) -> None:
    gc.collect()
    if device == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()


def command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL, timeout=15).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def metadata(args: argparse.Namespace) -> dict[str, Any]:
    info: dict[str, Any] = {
        "time": datetime.now().astimezone().isoformat(),
        "platform": platform.platform(), "macos": platform.mac_ver()[0],
        "machine": platform.machine(), "python": platform.python_version(),
        "python_executable": sys.executable, "torch": str(torch.__version__),
        "device": args.device, "cpu_threads": torch.get_num_threads(),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "previous_mps_fallback": _PREVIOUS_FALLBACK,
        "mps_environment": {k: v for k, v in os.environ.items() if k.startswith("PYTORCH_MPS_")},
        "mps_fallback_for_this_process": "0",
    }
    if platform.system() == "Darwin":
        info["chip"] = command_output(["sysctl", "-n", "machdep.cpu.brand_string"])
        info["memory_bytes"] = command_output(["sysctl", "-n", "hw.memsize"])
        # 只保存芯片/核心数，不保存显示器序列号等完整系统报告。
        try:
            displays = json.loads(command_output(["system_profiler", "SPDisplaysDataType", "-json"]))
            info["gpu_summary"] = [
                {k: v for k, v in entry.items() if k in {"sppci_model", "sppci_cores", "spdisplays_cores"}}
                for entry in displays.get("SPDisplaysDataType", [])
            ]
        except (ValueError, TypeError):
            info["gpu_summary"] = "unavailable"
    return info


def dtype_for(name: str) -> torch.dtype:
    dtype = getattr(torch, DTYPES[name], None)
    if dtype is None:
        raise NotImplementedError(f"当前 PyTorch 没有 {DTYPES[name]} 类型。")
    return dtype


def make_runner(shape: Shape, name: str, device: str, mode: str, route: str
                ) -> tuple[Callable[[], None], Callable[[], dict[str, Any]], dict[str, Any]]:
    """准备输入；只把 step 中的操作纳入计时。"""
    m, k, n = shape
    dtype = dtype_for(name)
    generator = torch.Generator(device="cpu").manual_seed(20260926)
    a0 = torch.randn(m, k, generator=generator, dtype=torch.float32) * 0.1
    b0 = torch.randn(k, n, generator=generator, dtype=torch.float32) * 0.1
    backward = mode == "fwd_bwd"
    output_dtype = torch.float32 if route == "bf16_to_fp32" else (
        torch.float16 if route == "fp8_cpu_to_fp16" else dtype)
    if route == "fp8_cpu_to_fp16":
        a, b = a0.to(dtype), b0.to(dtype)  # 预先量化为 CPU 上的真正 FP8 存储。
    else:
        a, b = a0.to(dtype).to(device), b0.to(dtype).to(device)
    out = torch.empty(m, n, dtype=output_dtype, device=device)
    g0 = torch.randn(m, n, generator=generator, dtype=torch.float32) * 0.1 if backward else None
    grad_output = g0.to(dtype).to(device) if g0 is not None else None
    if backward:
        a.requires_grad_(True)
        b.requires_grad_(True)

    def step() -> None:
        nonlocal out
        if backward:
            a.grad = None
            b.grad = None
            out = torch.mm(a, b)
            out.backward(grad_output)
        elif route == "bf16_to_fp32":
            # 两个转换也计时：这不是“BF16 原生算力”。
            torch.mm(a.float(), b.float(), out=out)
        elif route == "fp8_cpu_to_fp16":
            # CPU 解码 + 移到目标设备 + FP16 GEMM 全部计时。
            # 这是明确的兼容性路径，不是自动 fallback，更不是 FP8 GPU GEMM。
            aa = a.to(torch.float16).to(device)
            bb = b.to(torch.float16).to(device)
            torch.mm(aa, bb, out=out)
        else:
            # 前向测试复用输出，避免把每次分配输出的成本当作 GEMM 成本。
            torch.mm(a, b, out=out)

    def relative_error(actual: torch.Tensor, reference: torch.Tensor) -> float:
        actual = actual.detach().cpu().double()
        denominator = max(torch.linalg.vector_norm(reference).item(), 1e-30)
        return torch.linalg.vector_norm(actual - reference).item() / denominator

    def inspect() -> dict[str, Any]:
        # 与原始 FP32 输入的 CPU FP64 结果比较：包含输入量化、计算和输出舍入误差。
        # 校验在计时区外进行，只抽查左上 16x16 元素；不构成完整正确性证明。
        s = min(16, m, n, k)
        reference = a0[:s, :].double() @ b0[:, :s].double()
        validation: dict[str, Any] = {
            "sample_relative_l2_error": relative_error(out[:s, :s], reference),
            "all_output_finite": bool(torch.isfinite(out.detach().float()).all().item()),
            "sample_size": s,
            "reference": "original_FP32_inputs_computed_on_CPU_FP64",
            "tolerance": TOLERANCES[name],
        }
        errors = [validation["sample_relative_l2_error"]]
        finite = validation["all_output_finite"]
        if backward:
            assert g0 is not None and a.grad is not None and b.grad is not None
            da_ref = g0[:s, :].double() @ b0[:s, :].double().T
            db_ref = a0[:, :s].double().T @ g0[:, :s].double()
            validation["grad_a_sample_relative_l2_error"] = relative_error(a.grad[:s, :s], da_ref)
            validation["grad_b_sample_relative_l2_error"] = relative_error(b.grad[:s, :s], db_ref)
            validation["all_gradients_finite"] = bool(
                torch.isfinite(a.grad.detach().float()).all().item()
                and torch.isfinite(b.grad.detach().float()).all().item())
            errors += [validation["grad_a_sample_relative_l2_error"], validation["grad_b_sample_relative_l2_error"]]
            finite = finite and validation["all_gradients_finite"]
        validation["passed"] = bool(finite and all(math.isfinite(v) and v <= TOLERANCES[name] for v in errors))
        return validation

    detail = {"input_dtype": str(dtype), "output_dtype": str(output_dtype),
              "input_device": str(a.device), "compute_device": device,
              "accumulator_dtype": "not_inspected", "native_hardware_support": "not_proven"}
    return step, inspect, detail


def timed_block(step: Callable[[], None], device: str, iterations: int) -> float:
    sync(device)
    start = time.perf_counter()
    for _ in range(iterations):
        step()
    sync(device)  # 不等 GPU 完成，会误把命令提交时间当作计算时间。
    return time.perf_counter() - start


def benchmark(shape: Shape, name: str, mode: str, route: str, args: argparse.Namespace) -> dict[str, Any]:
    m, k, n = shape
    # 保守粗估：输入、输出、原始 CPU 输入、转换和梯度；不含驱动/JIT/并发临时内存。
    estimate = 4 * (m * k + k * n + m * n) * (6 if mode == "fwd_bwd" else 4)
    if estimate > args.max_estimated_gib * 1024**3:
        return {"status": "skipped_budget", "estimated_bytes": estimate}
    context = torch.enable_grad() if mode == "fwd_bwd" else torch.inference_mode()
    with context:
        step, inspect, detail = make_runner(shape, name, args.device, mode, route)
        for _ in range(args.warmup):
            step()
        sync(args.device)
        iterations = 1
        # 反向和转换会产生临时张量，因此使用更低的排队上限。
        max_iters = min(args.max_iters, 8) if mode == "fwd_bwd" or route != "direct" else args.max_iters
        while True:
            elapsed = timed_block(step, args.device, iterations)
            if elapsed >= args.target_ms / 1000 or iterations >= max_iters:
                break
            iterations = min(iterations * 2, max_iters)
        samples = [timed_block(step, args.device, iterations) / iterations for _ in range(args.repeats)]
        median_seconds = statistics.median(samples)
        flops = (6 if mode == "fwd_bwd" else 2) * m * k * n
        # 先保留计时；若后续校验算子失败，也不冒充该案例已通过正确性检查。
        result = {**detail, "iterations_per_repeat": iterations, "repeats": args.repeats,
                  "median_ms": median_seconds * 1000, "min_ms": min(samples) * 1000,
                  "max_ms": max(samples) * 1000, "samples_ms": [v * 1000 for v in samples],
                  "effective_tflops": flops / median_seconds / 1e12,
                  "flops_per_iteration": flops, "estimated_bytes": estimate}
        try:
            result["validation"] = inspect()
            result["status"] = "ok" if result["validation"]["passed"] else "numerical_warning"
        except Exception as exc:
            result["status"] = "validation_error"
            result["validation_error"] = f"{type(exc).__name__}: {exc}"
        return result


def classify_error(exc: Exception) -> str:
    text = str(exc).lower()
    if isinstance(exc, NotImplementedError) or any(token in text for token in (
        "not implemented", "not supported", "does not support", "doesn't support",
        "does not have support", "unsupported", "not support")):
        return "unsupported"
    return "failed"  # 内存不足、API 变化等不能一概说成“硬件不支持”。


def json_safe(value: Any) -> Any:
    """将 NaN/Inf 写成字符串，保证异常数值也能保存为合法 JSON。"""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def save(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = arguments()
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS 不可用；本脚本不会偷偷改用 CPU。请先检查环境。", file=sys.stderr)
        return 2
    shapes = args.shapes or [(s, s, s) for s in args.sizes]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = args.output or Path("runs") / f"{args.device}_benchmark_{stamp}.json"
    if path.exists():
        print(f"结果文件已存在，不覆盖：{path}", file=sys.stderr)
        return 2
    options = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    report: dict[str, Any] = {"schema_version": 1, "environment": metadata(args), "options": options,
                            "notes": ["Effective operator throughput, not hardware peak.",
                                      "Direct means no explicit cast in the timed Python code; internal accumulation is not inspected.",
                                      "Forward uses preallocated output; backward includes autograd and both operand gradients, no optimizer.",
                                      "Conversion routes report useful GEMM FLOPs / end-to-end time, not native low-precision FLOPs.",
                                      "Sample validation is not exhaustive."],
                            "probes": [], "results": [], "complete": False}
    save(path, report)
    print(f"device={args.device}; torch={torch.__version__}; Python={platform.python_version()}")
    print("自动 CPU fallback 已禁用。每项先做小矩阵探测，不支持时保留原因。")
    print("结果是有效吞吐量，不是芯片峰值；direct 也不证明原生指令。")
    print(f"JSON: {path}\n")
    jobs = [(d, mode, "direct") for d in args.dtypes
            for mode in (["fwd", "fwd_bwd"] if args.backward else ["fwd"])]
    if args.conversions:
        jobs += [("bf16", "fwd", "bf16_to_fp32"),
                 ("fp8_e4m3fn", "fwd", "fp8_cpu_to_fp16"),
                 ("fp8_e5m2", "fwd", "fp8_cpu_to_fp16")]
    try:
        for name, mode, route in jobs:
            label = f"{name}/{mode}/{route}"
            probe = {"dtype": name, "mode": mode, "route": route}
            try:
                context = torch.enable_grad() if mode == "fwd_bwd" else torch.inference_mode()
                with context:
                    step, _, _detail = make_runner((32, 32, 32), name, args.device, mode, route)
                    step()
                    sync(args.device)
                    del step, _detail, _
                probe["status"] = "ok"
            except Exception as exc:
                probe.update(status=classify_error(exc), error=f"{type(exc).__name__}: {exc}")
            report["probes"].append(probe)
            save(path, report)
            cleanup(args.device)
            if probe["status"] != "ok":
                print(f"SKIP {label}: {probe['error']}", flush=True)
                continue
            for shape in shapes:
                row: dict[str, Any] = {"dtype": name, "mode": mode, "route": route,
                                       "shape_m_k_n": list(shape)}
                try:
                    row.update(benchmark(shape, name, mode, route, args))
                except Exception as exc:
                    row.update(status=classify_error(exc), error=f"{type(exc).__name__}: {exc}")
                report["results"].append(row)
                save(path, report)
                dims = "x".join(map(str, shape))
                if "median_ms" in row:
                    error = row.get("validation", {}).get("sample_relative_l2_error")
                    error_text = f"{error:.3e}" if error is not None else "unchecked"
                    print(f"{row['status'].upper():17} {label:37} {dims:18} "
                          f"{row['median_ms']:9.3f} ms  {row['effective_tflops']:8.3f} effective TFLOP/s  "
                          f"rel_err={error_text}", flush=True)
                else:
                    print(f"{row['status'].upper()} {label} {dims}: {row.get('error', '超过估计内存预算')}", flush=True)
                cleanup(args.device)
        report["complete"] = True
    except KeyboardInterrupt:
        report["interrupted"] = True
        print("\n已中止。已完成的案例保留在 JSON 中。")
    finally:
        save(path, report)
    print(f"\n结果已保存：{path}")
    return 0 if report["complete"] else 130


if __name__ == "__main__":
    raise SystemExit(main())
