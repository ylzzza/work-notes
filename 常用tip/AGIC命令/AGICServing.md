---
tags:
  - AGIC
  - AGIC/命令
  - AGIC/AGICServing
  - AGIC/部署
  - work
aliases:
  - AGICServing命令
  - AGICServing
---

# AGICServing 命令

1. 创建虚拟环境
```bash
make venv
```

2. 安装
```
开发者：`make dev AGIC_BACKEND=db500` / `make dev AGIC_BACKEND=db530`
用户：`make install AGIC_BACKEND=db500` / `make install AGIC_BACKEND=db530`
```

```
3. 设置环境变量
```bash
source env_setup_agic.sh AGIC_BACKEND=db500
# 或
source env_setup_agic.sh AGIC_BACKEND=db530
```

```
 现在直接用 就好了
  export EXECUTION_ENVIRONMENT=2
make dev AGIC_BACKEND=db500
source env_setup_agic.sh AGIC_BACKEND=db500   AGIC_DUMP_DEBUG_MODE=2



AGIC_PROFILE_LEVEL=1 性能指令
```
4. 运行 Qwen 示例

## Offline 推理
```bash
python example/offline_inference/test_sim_npu.py \
    -m /path/to/qwen_model \
    --prompts "your prompt here"
```

> 其中 `-m` 指定模型路径，`--prompts` 指定输入 prompt。

## 分布式推理（以 DP2TP2EP4 为例）
```bash
cd example/offline_inference
python data_parallel_agic.py \
    --model=/path/to/qwen_model \
    --dp-size=2 \
    --tp-size=2 \
    --node-size=1 \
    --node-rank=0 \
    --master-addr=192.168.100.12 \
    --master-port=12345
```

## Online 推理

**后台启动 vLLM 服务**
```bash
vllm serve /root/share/qwen3_0.6b_packing &
vllm serve ../AGICDist/agicdist/qwen2_5_all/ &

# 或使用 nohup
nohup vllm serve ../AGICDist/agicdist/qwen2_5_all/ > run.log 2>&1 &
```

**发送 HTTP 请求**（等模型加载完成后）
```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/root/share/qwen3_0.6b_packing",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```


**如何使用AGICServing 编译模型**
```
make dev AGIC_BACKEND=db500
source env_setup_agic.sh AGIC_BACKEND=db500

# 使用一个新目录，确保重新编译而不是命中旧缓存
export VLLM_AGIC_CACHE_ROOT="$PWD/.cache/vllm_agic_rebuild"
export VLLM_AGIC_USING_CACHE=1

python example/offline_inference/test_sim_npu.py \
    --graph_mode \
    --model /root/share/yelinzhong/model/Qwen3-8b-l1 \
    --prompts "你好，请介绍一下你自己" \
    --max_new_tokens 5

--graph_mode 很关键：不加时是 eager 模式，不会走 AGIC graph 编译执行。

首次运行日志应该出现：

Compiling AGIC graph: ...
AGIC graph artifacts are compiled and executor sessions are ready

产物会放在：

  AGICServing/.cache/vllm_agic_rebuild/
  └── agic_graph_cache/
      └── Qwen3-8b-l1/
          └── rank_0_0/
              ├── model.embed_tokens/
              ├── model.layers.0/
              ├── model.norm/
              ├── lm_head/
              ├── gather_logits/
              └── weights/
```

> [!tip] 相关文档
> - [[AGIC 日常命令]] — 命令总索引
> - [[AGICDist]] — Serving 依赖 AGICDist 的子图编译产物
