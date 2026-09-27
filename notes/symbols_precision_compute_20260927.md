# 符号、数据类型与算力：P00/P01 旁注

记录日期：2026-09-27。内容依据本次对话中的教学代码及以下官方资料，不是仓库最新版代码审计。

## out=out

在 `torch.mm(a, b, out=buffer)` 中，左侧 `out` 是 API 的关键字参数名，右侧 `buffer`
是已分配的输出张量变量。写成 `out=out` 只是两者恰好同名，不是比较运算或把 out 赋给自己。
不传 out 时通常返回新结果张量；传 out 可复用输出存储，但不保证内部绝不申请临时工作区。
匹配形状、设备和 dtype；`torch.empty` 的旧内容未初始化，不应在填入结果前使用。
需要自动求导的计算不要直接照搬 out=；PyTorch 的 out 变体不支持所需梯度。[1][2]

## FP8 和 MPS

用户日志只证明当前 PyTorch 2.14.0/MPS 的两个 float8 dtype 直接路径被拒绝。
教学脚本先在 CPU 准备 FP8，再在每次计时中转 FP16、转到 MPS 并执行 FP16 mm。
因此不是 GPU 的 FP8 GEMM 成绩。不能进一步推定 GPU 绝不可能用整数位模式保存 FP8 编码，
或自定义 Metal kernel 绝不可能在 GPU 上解码；那是另一个需要实现与验证的方案。
MPS 张量设备转换发生在统一内存系统上，不宜直接描述成独显的 PCIe 搬运。

## TF32

TF32 = TensorFloat-32，是 NVIDIA Tensor Core 的一种低精度乘法计算方式，不是 TensorFlow 32。
在典型路径中，FP32 输入保持 32 位存储；乘法输入舍入为 8 位指数、10 位小数字段的精度，
保留约 FP32 的指数范围；乘积以 FP32 累加。FP32 小数字段为 23 位。
10/23 不包含正规数隐含的最高有效位；不能把 TF32 当作新的一般性 19 位存储 dtype。
因此存储类型是 float32 不保证每个内部乘法都用了完整 FP32 精度。此模式不应套用到 Apple MPS。[3][4]

## M/N/K 与张量轴

`A[M,K] @ B[K,N] -> C[M,N]`：M 为输出行数，N 为输出列数，K 为匹配并累加的维度。
这描述两个二维矩阵的三个尺寸数字，不是单个三维张量。[5]
M/N/K 不是必须存在英文全称的首字母缩写。BLAS 规范明确使用它们；早期 Fortran 的默认
I～N 整数规则展示了这组字母的传统用途，但不能据此断言数学记号起源于 Fortran。[6]

高维张量可以约定 `X[B,T,D]`：B=batch size，T=token positions/time steps，D=feature dimension。
这些是局部约定而非唯一国际标准。一个字母在别处可以有别的含义：例如 B 也可能是矩阵名。
一般可写 `X ∈ ℝ^(d1×d2×...×dr)`，r 为轴的数量，di 为第 i 根轴的长度。

## A ∈ ℝ^(M×K) 怎么读

“矩阵 A 属于 M 行 K 列的实矩阵集合。”
∈ 读作“属于”或“是……的元素”；ℝ 是实数集合，R 可联系 Real numbers。
字体为 double-struck/blackboard bold，中文常称双线体/黑板粗体；Unicode 把 U+211D 明确列为
Double-Struck Capital R，并注明代表实数集合。[7]
右上角 M×K 在这里描述集合中矩阵的形状，不是计算某个数 R 的幂。
数学实数与有限精度计算机存储并不相同；FP32 是对可表示数值范围内实数的近似表示方式。

## Qwen 的 A3B 与共享专家

A 联系 activated parameters；B=billion，即 10^9，中文十亿。
Qwen3-30B-A3B 的官方值是总参数 30.5B、激活 3.3B；各 MoE 层有 128 个专家，每 token 选 8 个，
不是此前示例中的两个。[8]
Qwen3 技术报告明确说其 MoE 不设置共享专家，但它仍有所有 token 使用的注意力等公共模块。
“没有 shared expert 分支”不等于“没有公共参数或只有专家”。[9]
Qwen3.5-35B-A3B 的模型卡在语言模型部分标注总 35B、激活 3B，MoE 为 8 routed + 1 shared。[10]
不能把 `35B-A3B` 的 35B 读成 A35B。不同命名中的激活数量常有取整及统计口径，核对模型卡、
嵌入/输出头、是否包含视觉模块等，不把名字当成精确 FLOP 审计结果。

## 厂商如何确定 TFLOP/s 和 TOPS

理论峰值可用“可执行单元数量 × 每时钟操作数 × 时钟频率”推导，不一定来自完整模型跑分。
一次标量 FMA/MAC 常按乘法和加法计两次 operation；若每周期指标已按 operation 计数，不能再乘 2。
NVIDIA 官方性能指南直接展示按 SM 数量、频率及每周期能力计算 A100 峰值的方法。[11]
实际微基准可以重复执行 FMA 或矩阵乘法以测可达吞吐；官方 Grace 指南有汇编 FMA 微基准案例。[12]
数字解释还依赖精度、累加精度、稠密/稀疏、频率/功率及统计单卡/集群等边界。
IEEE 754 规范数字格式与算术，不是所有厂商宣传算力统一使用的测速程序。[13]
MLPerf 等另行规定工作负载、质量与延迟/吞吐评价要求；不要把营销峰值自动当成此类测试结果。[14]

## 官方来源

[1] https://docs.pytorch.org/docs/2.14/generated/torch.mm.html
[2] https://docs.pytorch.org/docs/2.14/notes/out.html
[3] https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/
[4] https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/drive-os-tensorrt-developer-guide/advanced.html
[5] https://netlib.org/lapack/explore-html/dd/d09/group__gemm_ga1e899f8453bcbfde78e91a86a2dab984.html
[6] https://docs.oracle.com/cd/E19957-01/805-4939/z40007365fbc/index.html
[7] https://www.unicode.org/charts/nameslist/n_2100.html
[8] https://huggingface.co/Qwen/Qwen3-30B-A3B
[9] https://arxiv.org/html/2505.09388v1
[10] https://huggingface.co/Qwen/Qwen3.5-35B-A3B
[11] https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html
[12] https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/health-checks.html
[13] https://standards.ieee.org/ieee/754/6210/
[14] https://mlcommons.org/benchmarks/inference-datacenter/
