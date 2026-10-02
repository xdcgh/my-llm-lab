# 学习进度：my-llm-lab

计划版本：v1.0
本次核对日期：2026-10-01（Mac 本地时区 +08:00）。
审阅依据：Desktop Commander Remote 实际读取本地工作区、Git 状态、课程代码和运行文件。
同步前本地 HEAD：`b9aa975dd301bb2b5e977c7bc1574d5ee2a88c73`，分支 `main`。
当前阶段：**P01 一个参数怎么学习**
当前小节：**P01.3 更新与清零：手写 20 步已有运行截图；学习率对照待运行与核对**。

## 证据与边界

本次已通过插件连接 `Mac-Studio.local`，读取 `/Users/xudacheng/Downloads/my-llm-lab`。
依据是本机已保存文件，不依赖是否已 commit/push；编辑器中未保存内容不在本次读取范围。
未执行 fetch/pull，也未核验 GitHub 的最新状态；本地 origin/main 只是已有远端跟踪信息。
“代码存在”“运行记录存在”“用户手写完成”“概念验收通过”分别记录。

## 当前环境

| 项目 | 已记录值 | 依据 |
|---|---|---|
| 硬件 | Apple M1 Ultra，GPU 48 核，64 GiB | 三份 MPS JSON 的 environment |
| macOS / 架构 | 27.2 / arm64 | 同上 |
| Python | 3.12.9，项目 `.venv` | 同上及 P01 JSON |
| PyTorch | 2.14.0 | 同上及 P01 JSON |
| NumPy | uv.lock 锁定 2.5.3；用户先前日志显示已安装 | uv.lock / 对话证据 |
| 环境管理 | uv；不改动现有 oMLX 环境 | 项目约定 |
| MPS | built / available 均为 true；三组矩阵测试 complete=true | 三份 MPS JSON |
| 当前课程设备 | CPU / FP32 | P01 源码与 JSON |

## 已有运行证据

P00：`runs/mps_benchmark_20260926_*.json` 共三份；分别记录前向、包含反向及显式转换路径。
其结果是当前算子路径的有效性能，不是芯片峰值或原生指令支持证明。

P01：`runs/p01_one_parameter_20260927_105647_955412.json` 已有六轮结果，配置为
`x=2, target=6, initial_w=1, learning_rate=0.1, steps=6`。
最后 `w=2.9998719692230225`，最后 loss 为 `6.556751941388939e-08`。
以上六步 JSON 属于参考实现，不是手写 20 步的日志。

手写实现：`lessons/mycode/train_one_parameter.py`，本次实际读取到 35 行，循环为 20 步。
当前对话截图显示用户已运行 step=0 至 step=19；源码与截图中的计算配置一致。
该脚本只将 records 留在内存并打印，未保存独立 JSON；本次没有代用户重跑训练。
学习率脚本 `lessons/p01_learning_rates.py` 已存在；当前 runs/ 未发现对应 JSON。
因此只记录为“代码已保存、实验结果待提供”，不据此断言用户绝未运行过。

助手另在隔离的 Linux CPU 环境（Python 3.13.5 / PyTorch 2.10.0+cpu）重跑快照中的
原课程脚本，六步正常完成，逐步梯度与解析式一致、loss 下降。它不是用户 Mac 的新增实测。

## 当前文件

- 参考实现：`lessons/p01_train_one_parameter.py`。
- 单次更新例子：`lessons/02_one_parameter.py`。
- 实际手写文件：`lessons/mycode/train_one_parameter.py`；保留其原有路径。
- 学习率对照：`lessons/p01_learning_rates.py`；不要机械使用旧说明中的 mycode 路径。
- 本地同步记录：`notes/learning_sessions/2026-10-01-local-sync.md`。
- 历史快照审阅说明：`notes/learning_sessions/2026-09-27-p01-review.md`。

## 理解与待核对问题

用户正在巩固“预测 → loss → 梯度 → 更新”的关系，不将解释过或运行过自动记为掌握。
重点核对：

1. `w.grad = None` 只清掉旧梯度，为何不把 w 重设为 1？
2. `loss.backward()` 后参数是否已经改变？真正更新参数的是哪一行？
3. `old_w = w.item()` 与 `old_w = w` 为什么不同？
4. 更新后为何重新计算 new_loss，旧 loss 为什么不会自动刷新？
5. `with torch.no_grad()` 为什么不等于冻结参数？

先前已讲解：返回值接口与输出参数接口、FMA、稠密/稀疏运算及厂商支持范围。
这些记录为“已讲解”，尚无用户逐项理解确认。

## 下一步唯一执行任务

阅读并核对本机已有 `lessons/p01_learning_rates.py` 的双层循环，在项目根目录运行：

```bash
uv run python lessons/p01_learning_rates.py
```

只比较学习率 0.01、0.1、0.2、0.25、0.3；每组重新从 w=1 开始，CPU/FP32，20 步。
运行后读取新产生的 runs/p01_learning_rates_*.json，比较收敛、振荡、发散和分段耗时。
脚本文档字符串仍示例为 lessons/mycode/ 路径，但文件实际在 lessons/；本次只纠正文档入口，不改代码。
完成结果与概念核对后再讲优化器封装；不提前开始 P02，不切换框架或设备。
梯度为何累加、参数与梯度的区别、学习率现象仍需用户确认，不能将讲解记为掌握。

## 历史保留

2026-09-27 快照审阅更新前的 PROGRESS 原文保存在
`notes/learning_sessions/2026-09-27-progress-before-d4f6a03-review.md`。
它保留初始化及上一轮追加内容，不再作为当前状态。

每轮按 notes/REPO_WORKFLOW.md 接续：插件可用时优先读本地工作区；GitHub 作为版本备份与后备读取方式。
未 push 的内容可通过已授权的本地工具读取，但不能视为 GitHub 上已有。
同步记录不等于自动提交、上传或后台监控；只有实际写入并回读后才能报告已更新。
本次修订前的文档保存在 .git/llm-lab-updates/ 对应时间戳备份中。
