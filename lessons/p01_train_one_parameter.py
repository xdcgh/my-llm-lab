"""P01：一个参数，连续更新六次。默认 CPU，不联网，不修改进度文件。"""
import json
import platform
from datetime import datetime
from pathlib import Path

import torch

# 输入、目标不需要梯度；只有 w 是要学习的参数。
x = torch.tensor(2.0, dtype=torch.float32)
target = torch.tensor(6.0, dtype=torch.float32)
w = torch.tensor(1.0, dtype=torch.float32, requires_grad=True)
learning_rate = 0.1
records = []

# 核心训练循环。range(6) 让 step 依次取 0、1、2、3、4、5。
for step in range(6):
    w.grad = None                         # 清掉上一轮梯度，不改变 w。
    prediction = x * w                     # 前向：用当前参数预测。
    loss = (prediction - target) ** 2      # 前向：计算平方误差。
    loss.backward()                       # 反向：求 loss 对 w 的梯度。
    old_w = w.item()                       # 取普通 Python 数值，供记录。
    old_loss = loss.item()
    gradient = w.grad.item()

    with torch.no_grad():                 # 参数更新不记录新的求导关系。
        w -= learning_rate * w.grad       # 真正修改参数。
        new_loss = ((x * w - target) ** 2).item()  # 必须重新计算新 loss。

    records.append({"step": step, "w_before": old_w, "loss_before": old_loss,
                    "gradient": gradient, "w_after": w.item(), "loss_after": new_loss})
    print(f"step={step} w={old_w:.6f} loss={old_loss:.6f} grad={gradient:.6f}"
          f" -> w={w.item():.6f} new_loss={new_loss:.9f}")

# 以下只是实验存档，不属于模型计算。无需在这一课逐行学习文件 API。
# 存档位置相对本脚本的项目根目录，不依赖终端当前目录。
run_dir = Path(__file__).resolve().parents[1] / "runs"
run_dir.mkdir(parents=True, exist_ok=True)
stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
path = run_dir / f"p01_one_parameter_{stamp}.json"
report = {"lesson": "P01", "device": "cpu", "torch_version": str(torch.__version__),
          "python_version": platform.python_version(),
          "config": {"x": 2.0, "target": 6.0, "initial_w": 1.0,
                     "learning_rate": learning_rate, "steps": 6},
          "records": records,
          "note": "运行记录不是理解验收；请另行确认是否理解梯度清零和参数更新。"}
with path.open("x", encoding="utf-8") as file:
    json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
print(f"实验记录：{path.name}（项目 runs/ 目录）")
