# 学习进度：my-llm-lab

计划版本：v1.0
本次核对日期：2026-09-27
审阅依据：用户上传的 Git 归档，归档注释提交为 `d4f6a03e0c8ec49de1e8aa7468ee683be31b8b57`。
当前阶段：**P01 一个参数怎么学习**
当前小节：**P01.3 更新与清零：手写六次训练循环并核对理解**。

## 证据与边界

本次已能打开公开 GitHub 仓库首页；具体文件页、raw/API 抓取仍失败，因此逐行审阅以
上传快照为准，尚未独立核对当前远端 `main` 是否仍指向上述提交。
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
这不表示用户已完成手写或通过概念验收。

助手另在隔离的 Linux CPU 环境（Python 3.13.5 / PyTorch 2.10.0+cpu）重跑快照中的
原课程脚本，六步正常完成，逐步梯度与解析式一致、loss 下降。它不是用户 Mac 的新增实测。

## 当前文件

- 参考实现：`lessons/p01_train_one_parameter.py`。
- 单次更新例子：`lessons/02_one_parameter.py`。
- 手写目标：`lessons/p01_handwritten.py`，本次补丁**不创建**该文件，留给用户手写。
- 本轮审阅说明：`notes/learning_sessions/2026-09-27-p01-review.md`。

## 理解与待核对问题

用户正在巩固“预测 → loss → 梯度 → 更新”的关系，不将解释过或运行过自动记为掌握。
重点核对：

1. `w.grad = None` 只清掉旧梯度，为何不把 w 重设为 1？
2. `loss.backward()` 后参数是否已经改变？真正更新参数的是哪一行？
3. `old_w = w.item()` 与 `old_w = w` 为什么不同？
4. 更新后为何重新计算 new_loss，旧 loss 为什么不会自动刷新？
5. `with torch.no_grad()` 为什么不等于冻结参数？

本轮答疑：返回值接口与输出参数接口、FMA、稠密/稀疏运算及厂商支持范围。
这些记录为“已讲解”，尚无用户逐项理解确认。

## 下一步唯一执行任务

用户手写同一单参数训练循环，建议另存 `lessons/p01_handwritten.py`，
运行 `uv run python lessons/p01_handwritten.py`，提交并推送代码与实际输出。
保留原参考实现与已有运行记录；不要开始 P02 或同时更换模型、数据与框架。

手写核对后，沿 ROADMAP 的 P01.3 只改变学习率做对照，再理解优化器封装。
之后才进入 P02.1 字符编码。改变 x、target、初值或轮数时，注意原参考实现的
JSON config 中有写死的值，必须同步记录；本次补丁不修改该代码。

## 历史保留

本次更新前的 PROGRESS 原文保存在
`notes/learning_sessions/2026-09-27-progress-before-d4f6a03-review.md`。
它保留初始化及上一轮追加内容，不再作为当前状态。

每轮以最新实际证据更新此文件，并保持：未 push 的本地内容不能视为远端可见；
无论何种工具，未实际写入就不能说已更新用户本机或 GitHub 文件。
