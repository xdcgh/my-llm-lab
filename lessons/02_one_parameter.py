import torch

w = torch.tensor(1.0, requires_grad=True)
prediction = 2 * w
loss = (prediction - 6) ** 2

print(f"更新前：w={w.item():.4f}, loss={loss.item():.4f}")
print("backward 之前 w.grad:", w.grad)

loss.backward()
print(f"backward 之后 w={w.item():.4f} w.grad={w.grad.item():.4f}")

with torch.no_grad():
    w -= 0.1 * w.grad
    new_prediction = 2 * w
    new_loss = (new_prediction - 6) ** 2

print(f"更新后：w={w.item():.4f}, loss={new_loss.item():.4f}")
