# P01 学习记录：手写循环已运行，下一步只改变学习率

记录日期：2026-09-27。依据是用户本轮提供的两张截图，不是对本地完整源码或远端最新提交的审阅。

## 已有证据

- 用户手写文件位于 `lessons/mycode/train_one_parameter.py`。
- 截图可见 `x=2`、`target=6`、`w` 初始为 1、FP32、学习率 0.1、20 次更新。
- 循环保留每次更新前清除梯度、前向计算、反向传播与 no_grad 下参数更新。
- 终端已输出 step=0 至 step=19；参数逐渐接近 3，后段输出显示为零的 loss 和梯度。
- 截图使用固定小数位打印。显示为零不总等于存储值为零；需用科学计数法或精确比较区分。本例 CPU/FP32 也可能在后期舍入到可精确表示的 w=3，随后 loss 真正为零。
- 用户仍在理解：梯度为什么默认累加；小计算为何可能等待；Git 分页查看器的退出方式。

## 本次答疑

- Git 的差异内容由分页查看器（通常是 less）显示时，点击终端按小写 q 退出；Ctrl+C 不必然退出 less。
- 后续查看可以使用 `git --no-pager diff`，不必修改全局配置。
- `backward()` 把新梯度累积到 `.grad`。清除梯度的时机由训练循环决定。
- 标准微批梯度累积：先清梯度，保持参数不变，处理多个小批量并反向，按正确权重归一化，最后统一更新一次。
- 不要把每次已更新参数后的旧梯度直接累加，误认为这就是标准的大批量模拟；它也不是优化器动量的同义词。
- 当前是单标量 CPU 教学程序。仅凭等待感无法判断瓶颈；框架导入、初始化、终端输出和设备调度等都应与真正算术分开。

## 下一步唯一主实验

新建 `lessons/mycode/p01_learning_rates.py`，保留原手写文件。运行：

```bash
uv run python lessons/mycode/p01_learning_rates.py
```

保持数据、初始参数、20 次更新、CPU/FP32、清梯度规则不变，仅比较学习率：

`0.01, 0.1, 0.2, 0.25, 0.3`

每组必须重新从 w=1 开始。每组内部只清除旧梯度，不把 w 重新设成 1。

该脚本额外记录 torch 导入耗时、含打印的循环耗时与 runs/ 下的 JSON；这不是硬件纯算力基准，也不包括脚本进入前的 uv/Python 启动耗时。

## 待用户核对

1. 0.25 时是否在 w=1 和 w=5 间来回，loss 保持 16？
2. 0.3 时是否越过目标并越来越远？
3. 0.2 虽然比 0.1 更大，为什么在本例里不一定更快？
4. 等待主要发生在第一行输出之前、torch 导入阶段，还是训练循环阶段？

这里的临界值只属于 `loss=(2w-6)**2` 和普通手动梯度下降，不可当成所有模型的通用学习率。

## 接续边界

- 当前已确认“手写程序有运行输出”，不等同于全部概念已掌握。
- 新学习率实验尚未收到用户反馈，不标为完成。
- 下一阶段在核对结果后讲优化器封装；暂不同时更换设备、模型、数据或精度。
- 本轮网页工具可以读取公开仓库首页，但未取得最新具体文件或提交哈希；不以可能缓存的首页替代最新状态。
- 不覆盖 PROGRESS.md、README.md 或其他旧文件；用户可将本记录保存进 `notes/learning_sessions/`，并在确认后自行提交和推送。

## 本脚本本地验证

助手已在 Linux CPU、PyTorch 2.10.0+cpu、Python 3.13 环境运行五组对照，检查了 100 条记录、每组独立初始化、梯度公式与收敛/振荡/发散趋势。不是用户 Mac 的速度测试，用户具体数值与耗时以其实际输出为准。

## 官方参考

- less FAQ（q 与 Ctrl+C）：https://greenwoodsoftware.com/less/faq.html
- Git --no-pager：https://git-scm.com/docs/git
- PyTorch 梯度清除：https://docs.pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html
- PyTorch 梯度累积示例：https://docs.pytorch.org/docs/2.14/notes/amp_examples.html#gradient-accumulation
- PyTorch Benchmark：https://docs.pytorch.org/tutorials/recipes/recipes/benchmark
