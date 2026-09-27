import torch

x = torch.tensor(2.0, dtype=torch.float32)
target = torch.tensor(6.0, dtype=torch.float32)
w = torch.tensor(1.0, dtype=torch.float32, requires_grad=True)
learning_rate = 0.1
records = []

for step in range(20):
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

    records.append(
        {
            "step": step,
            "w_before": old_w,
            "loss_before": old_loss,
            "gradient": gradient,
            "w_after": w.item(),
            "loss_after": new_loss,
        }
    )
    print(
        f"step={step} w={old_w:.6f} loss={old_loss:.6f} grad={gradient:.6f}"
        f" -> w={w.item():.6f} new_loss={new_loss:.9f}"
    )
