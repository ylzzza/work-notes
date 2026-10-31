---
tags:
  - AGIC
  - AGIC/命令
  - AGIC/AGICDist
  - work
aliases:
  - AGICDist命令
  - AGICDist
---

# AGICDist 命令

> [!info] 不需要进入模拟器

1. 环境设置
```bash
make venv
```

2. 安装项目依赖并选择后端
   - 用户模式：`make install AGIC_BACKEND=db500` / `make install AGIC_BACKEND=db530`
   - 开发模式：`make dev AGIC_BACKEND=db500` / `make dev AGIC_BACKEND=db530`

3. 激活环境变量、准备 AGICDist 的开发/运行环境
```bash
source env_setup.sh
```

4. 进入工作目录
```bash
cd agicdist
```

5. 创建目录
```bash
mkdir -p 目录 && cd 目录
```

6. 子图编译
```bash
python ../run_submodules.py -c /ai/db500_data/nn_data/Qwen2.5-0.5B
```

> [!tip] 相关文档
> - [[AGIC 日常命令]] — 命令总索引
> - [[AGICServing]] — Serving 依赖 AGICDist 的子图编译产物
