"""把 (x * x).sum() 的前向计算与梯度分开观察；不更新任何参数。
运行：uv run python lessons/01_gradient_explained.py
本节使用 CPU；FP64 用于稳定显示很小的差分，不放到 MPS 上。
"""
import torch

x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
squares = x * x
value = squares.sum()

print("输入 x:", x.detach().tolist())
print("各自平方:", squares.detach().tolist())
print("累加结果 value:", value.item())
print("backward 之前 x.grad:", x.grad)
value.backward()
print("backward 之后 x.grad:", x.grad.tolist())
print("x 本身并未改变:", x.detach().tolist())

# 只改变一个输入，观察输出变化 / 输入变化；这是验证，不是 autograd 的实现。
epsilon = 0.001
for i in range(3):
    perturbed = x.detach().clone()
    perturbed[i] += epsilon
    new_value = (perturbed * perturbed).sum()
    rate = (new_value.item() - value.item()) / epsilon
    print(f"只把 x[{i}] 增加 {epsilon}: value={new_value.item():.6f}, 变化率≈{rate:.6f}")

# 换一个公式，就会得到不同梯度；梯度不是输入固定乘 2。
z = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
z.sum().backward()
print("改成 z.sum()，梯度变成:", z.grad.tolist())
