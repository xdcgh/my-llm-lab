# P00.1：只读检查 Mac 环境

## 本节目标

知道“当前终端调用的到底是哪一个 Python”。先不安装 PyTorch，不下载模型，不开始训练，也不更改现有 oMLX 环境。

在 Mac 的终端中运行：

```bash
uname -m
sw_vers -productVersion
python3 --version
python3 -c "import platform, sys; print('Python路径:', sys.executable); print('Python架构:', platform.machine())"
sysctl -n hw.memsize
```

## 每行是什么意思

| 命令 | 读取的信息 | 我们关注什么 |
|---|---|---|
| `uname -m` | 当前进程环境看到的机器架构 | Apple Silicon 原生环境通常应是 `arm64` |
| `sw_vers -productVersion` | macOS 版本 | 用于核对训练框架的系统要求 |
| `python3 --version` | 终端默认 Python 的版本 | 判断能否用于新建学习环境 |
| `python3 -c ...` | 实际 Python 可执行文件路径及进程架构 | 防止终端是 ARM，却调用了不同架构的 Python |
| `sysctl -n hw.memsize` | 内存大小，单位为字节 | 核对硬件预算；64 GiB 对应 68719476736 字节 |

`-c` 表示让 Python 直接执行后面的短代码。`import` 导入模块，`sys.executable` 是当前解释器位置，`platform.machine()` 返回进程所看到的架构。

这里没有安装、删除、重启服务或上传操作。不要额外运行 `sudo pip install` 或更改系统保护设置。输出中的本机用户名可以遮去，保留路径结构即可。

## 遇到报错

若 `python3` 不存在、系统弹出开发工具安装提示，或输出架构为 `x86_64`，先停在这里，把现象和输出交给助手，不重复盲装依赖。已有 Python 版本较旧也不是问题，下一节再选独立环境方案。

## 完成标准

五条命令的输出已获取并核对。**这仅完成环境信息收集，不等于 PyTorch 或 MPS 已经安装/验证。**

下一小节：P00.2，建立专属虚拟环境。然后 P00.3 才实际检查 PyTorch 的 CPU/MPS 前向、反向和更新。
