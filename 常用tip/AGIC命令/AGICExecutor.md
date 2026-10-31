---
tags:
  - AGIC
  - AGIC/命令
  - AGIC/AGICExecutor
  - AGIC/测试
  - work
aliases:
  - AGICExecutor命令
  - AGICExecutor
---

# AGICExecutor

> [!warning] 需要先登录模拟器
> 参见 [[基础环境#登录模拟器]]

## 日志配置

```bash
export AGRT_LOG_LEVEL=1
```

## 编译

**Python 版本**
```bash
source agic_executor.sh
```

**C++ 版本**
```bash
source env.sh
./build.sh clean  # 可不执行
./build.sh
```

## 运行

**Python 版本**
```bash
source env.sh
cd agic_python
python test.py --ckpt /your/path/qwen2.5_0.5b
```

**C++ 版本**
```bash
./agicexecutor --ckpt /your/path/qwen2.5_0.5b
```

## UT
```bash
./build/agic_test
```

## 单算子精度测试

### 环境准备
```bash
cd /localdata/env/sim_npu/share/yelinzhong/AGICExecutor
source env.sh
```

### C++ 单算子测试

统一入口 `tests/test.cpp`，支持的 suite：
```bash
./build/agic_test gemm
./build/agic_test rmsnorm
./build/agic_test mlp
./build/agic_test attn_online
./build/agic_test embedding
./build/agic_test gather
./build/agic_test topk
./build/agic_test moe_mlp_gate
```

自动设置环境变量：
```cpp
setenv("CHECK_SIMILARITY", "1", 1);
setenv("TILE_NUM", "48", 1);
```

Golden 数据路径：
```text
/ai/db500_data/for_agicexecutor_test/models/build_qwen3_0.6b/outputs/step0_prefill_npu/
```

日志中的精度对比结果：
```text
Cosine Similarity: ...
MSE: ...
```

### Python 单算子测试

> [!tip] 调试推荐用 Python 脚本

```bash
cd /localdata/env/sim_npu/share/yelinzhong/AGICExecutor
source env.sh
export PYTHONPATH=$PWD/agic_executor:$PYTHONPATH

python tests/python/gemm_test.py
```

指定 ckpt 和 GT：
```bash
python tests/python/gemm_test.py \
  --ckpt /ai/db500_data/for_agicexecutor_test/models/build_qwen3_0.6b/ \
  --gt-path /ai/db500_data/for_agicexecutor_test/models/build_qwen3_0.6b/outputs/step0_prefill_npu/ \
  --token-len 16 \
  --tile-num 48
```

可用脚本（`tests/python/`）：
```text
gemm_test.py
rmsnorm_test.py
mlp_test.py
attention_online_test.py
embedding_test.py
gather_test.py
topk_test.py
```

### 新增算子测试

参考现有实现：
- C++：`tests/src/ops/gemm_tests.cpp`
- Python：`tests/python/gemm_test.py`

关键配置：
```python
os.environ["CHECK_SIMILARITY"] = "1"
os.environ["GT_PATH"] = gt_path
```

算子 `forward()` 调用 `check_similarity(out_data)` 后，自动对比：
```text
$GT_PATH/<op_name>.io.npz
$GT_PATH/<op_name>.output.npz
```

> [!tip] 相关文档
> - [[AGIC 日常命令]] — 命令总索引
> - [[基础环境]] — 登录模拟器、SDK 路径
