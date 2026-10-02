# my-llm-lab

## 当前入口（2026-10-01 本地连接核对后更新）

P00 已有运行证据；手写 `lessons/mycode/train_one_parameter.py` 的 20 次更新已有对话截图。
当前 P01.3：学习率对照脚本已保存为 `lessons/p01_learning_rates.py`，运行结果待核对。
从项目根目录执行 `uv run python lessons/p01_learning_rates.py`；不按旧示例误用 mycode 路径。
请以 `PROGRESS.md` 为当前摘要；跨聊天入口为 `HANDOFF.md`。
插件可用时直接读取已保存的本地项目，无需先 push；GitHub 继续承担版本备份与后备读取。
本机接续协议见 `notes/REPO_WORKFLOW.md`；代码已运行不等于概念已掌握。

以下“起步包”说明是最初交付时的历史介绍，不代表仍在等待首次环境检查。

---

# 初始起步包说明（历史）

这是一个逐行学习大模型的长期项目计划，版本 v1.0（2026-09-26），不是已经训练好的模型，也不是一键训练脚本。

## 先看什么

打开 `ROADMAP.md` 了解路线，再读 `PROGRESS.md` 确认当前状态。当前从 `lessons/P00_environment.md` 开始：只读检查 Mac 环境，将输出用于下一节安装决策。

建议保存到个人代码目录，例如 `~/Code/my-llm-lab`。已有同名项目时保留原文件，使用新目录，不覆盖旧代码。这个起步包没有安装或启动操作。

## 包内文件

| 文件 | 用途 |
|---|---|
| ROADMAP.md | 从零训练到 Agent 后训练的完整阶段计划 |
| PROGRESS.md | 实際进度；当前尚未收到 Mac 执行结果 |
| HANDOFF.md | 换聊天的交接提示词 |
| AGENTS.md | 逐行教学与代码协作约定 |
| lessons/P00_environment.md | 当前唯一执行任务及逐行解释 |
| notes/README.md | 后续概念与问题笔记目录 |
| runs/README.md | 后续本地实验记录说明 |

Markdown 是普通文本，可用文本编辑器或代码编辑器打开。需要新聊天时，附上最新进度、相关代码和输出即可；不要假设模型能自动读取本机目录。

参考资料原始入口收录在 ROADMAP.md 末尾。
