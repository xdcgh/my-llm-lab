"""P01.3：只改变学习率；每组从 w=1 开始，CPU/FP32 训练 20 次。

从项目根目录运行：
    uv run python lessons/mycode/p01_learning_rates.py

这是学习代码，不是 CPU/GPU 性能基准。只新增一份 runs/ 下的 JSON，
不改原课程、进度文件或 Git 状态；不联网、不启动后台进程。
"""
from time import perf_counter

print("已经进入 Python 脚本；开始导入 PyTorch。", flush=True)
import_started = perf_counter()
import torch
import_seconds = perf_counter() - import_started
print(f"导入 torch 用时：{import_seconds:.4f} 秒", flush=True)

# 这些标准库只用于保存记录，不参与梯度计算。
import json
import platform
from datetime import datetime
from pathlib import Path

# 每组实验唯一改变的是学习率；数据、初始参数、精度与轮数都相同。
device = torch.device("cpu")
x = torch.tensor(2.0, dtype=torch.float32, device=device)
target = torch.tensor(6.0, dtype=torch.float32, device=device)
initial_w = 1.0
steps = 20
learning_rates = [0.01, 0.1, 0.2, 0.25, 0.3]
records = []
summaries = []
print(f"device={device}; dtype={x.dtype}; torch={torch.__version__}")

loop_started = perf_counter()
for learning_rate in learning_rates:
    # 新学习率 = 新实验，所以重新创建参数，避免继承上一组的训练结果。
    w = torch.tensor(initial_w, dtype=torch.float32, device=device, requires_grad=True)
    print(f"\n===== learning_rate={learning_rate} =====")

    for step in range(steps):
        # 同一组内部，只清除旧梯度，不重置已经学到的参数。
        w.grad = None
        prediction = x * w
        loss = (prediction - target) ** 2
        loss.backward()

        old_w = w.item()
        old_loss = loss.item()
        gradient = w.grad.item()

        with torch.no_grad():
            w -= learning_rate * w.grad
            new_loss = ((x * w - target) ** 2).item()

        records.append({
            "learning_rate": learning_rate, "step": step,
            "w_before": old_w, "loss_before": old_loss, "gradient": gradient,
            "w_after": w.item(), "loss_after": new_loss,
        })
        print(
            f"step={step:02d} w={old_w:.9f} loss={old_loss:.3e} "
            f"grad={gradient:.3e} -> w={w.item():.9f} new_loss={new_loss:.3e}"
        )

    summaries.append({
        "learning_rate": learning_rate,
        "final_w": w.item(), "final_loss": new_loss,
    })
loop_seconds = perf_counter() - loop_started

print("\n===== 每组 20 次更新后的汇总 =====")
for result in summaries:
    print(
        f"lr={result['learning_rate']:<4} "
        f"final_w={result['final_w']:.9f} final_loss={result['final_loss']:.6e}"
    )
print(f"训练循环用时（含记录和终端输出）：{loop_seconds:.6f} 秒")
print("上面两个计时都不包含执行本脚本之前的 uv/Python 启动耗时。")

# 存档附录：先理解上面的双层循环，再读这一段。
now = datetime.now().astimezone()
report = {
    "schema_version": 1, "created_at": now.isoformat(),
    "environment": {
        "python": platform.python_version(), "torch": str(torch.__version__),
        "platform": platform.platform(), "device": str(device), "dtype": str(x.dtype),
    },
    "config": {
        "input": x.item(), "target": target.item(), "initial_w": initial_w,
        "steps_per_experiment": steps, "learning_rates": learning_rates,
        "prediction": "x*w", "loss": "(prediction-target)**2",
        "update": "w -= learning_rate * w.grad", "reset_grad_each_update": True,
    },
    "timings_seconds": {"torch_import": import_seconds, "loop_including_logging": loop_seconds},
    "summaries": summaries, "records": records,
}
output_dir = Path("runs")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"p01_learning_rates_{now.strftime('%Y%m%d_%H%M%S_%f')}.json"
with output_path.open("x", encoding="utf-8") as handle:
    json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    handle.write("\n")
print(f"实验记录已保存：{output_path}")
