# 只在本地建立 Git，不关联 GitHub

在现有项目根目录执行：

```bash
git --version
git rev-parse --show-toplevel
```

输出当前 my-llm-lab 根目录：已经是仓库，不必再初始化。
输出上级目录：当前处于另一个仓库内，先核对边界，不要直接把上级目录内容加入提交。
报 `fatal: not a git repository`：没有找到仓库，继续：

```bash
git init -b main
```

编辑已有 .gitignore，缺少的条目补上，不覆盖已有内容：

```gitignore
.venv/
__pycache__/
*.py[cod]
.DS_Store
.env
.env.*
!.env.example
*.pt
*.pth
*.ckpt
*.safetensors
data/raw/
data/cache/
```

不要忽略 pyproject.toml、uv.lock、.python-version 或学习代码。
训练大文件不入 Git；小型基准 JSON 和学习笔记可以随代码保存。本地 Git 不等于异地备份。
.gitignore 不会自动移除已经被跟踪的文件。

补好依赖、确认环境运行后，建立首个本地快照：

```bash
git add .gitignore .python-version pyproject.toml uv.lock lessons/ *.md
git diff --cached --stat
git diff --cached --name-only
```

检查没有密钥、私密数据、.venv 和模型权重，再执行：

```bash
git commit -m "chore: initialize uv project and verify MPS"
git status --short
```

若提示 Author identity unknown，设置当前项目的作者信息（第二行邮箱必须替换）：

```bash
git config --local user.name "xudacheng"
git config --local user.email "替换成你打算用于Git提交的邮箱"
```

然后重新执行 commit。作者邮箱会保存在提交历史里；不是 GitHub 登录信息，也不会因此上传代码。
不执行 remote add / push，也不需要现在创建 GitHub 仓库。

参考：
- https://git-scm.com/docs/git-init
- https://git-scm.com/docs/git-rev-parse
- https://git-scm.com/docs/gitignore
- https://git-scm.com/book/en/v2/Customizing-Git-Git-Configuration
