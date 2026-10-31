
一句话概括：`Operator` 是所有 AGIC 算子的公共执行框架，负责解析编译产物、管理各内存 Zone、绑定 Kernel 参数、启动 NPU Kernel，并在执行后进行精度检查和数据 dump；具体算子类只需要组织自己的输入、输出和特殊参数。

主流程可以理解为：

```
构造 Operator
   ↓
读取 agic_module.json
   ↓
规划 inst/config/offline 内存
   ↓
load_state_dict() 加载指令、配置和权重
   ↓
具体算子分配输出和临时空间
   ↓
set_custom/set_weight/set_online 绑定参数
   ↓
Operator::forward()
   ↓
agicLaunchKernel + StreamSynchronize
   ↓
check_similarity / dump_io_data
   ↓
释放输入、online 临时空间
```

## 1. 构造阶段

基类构造函数位于 [operator.cpp (line 30)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:30)。

```
Operator::Operator(string ckpt_dir, string op_name)
```

主要读取算子目录下的：

```
agic_module.json
```

从中取得：

- `var_infos`：Kernel 参数名称到参数槽位的映射
- `custom_offsets`：输入输出张量的绑定顺序
- `online_offsets`：中间临时 Buffer 的布局
- `load_infos`：指令、配置、权重文件的信息
- `input_output_infos`：输入输出的 shape、dtype、动态 token 维度
- `extra_infos`：算子附加配置

例如：

```
input → arg slot 0
weight → arg slot 1
output → arg slot 2
token_len → arg slot 3
```

### Cache 模式

如果算子共享目录旁边存在 `weights/`，就进入 cache 模式：

```
agic_graph_cache/<model>/rank_0_0/
├── model.layers.mlp/
└── weights/
```

共享图目录保存指令，具体层权重从：

```
<artifact_root>/weights/
```

加载。

## 2. 内存 Zone 规划

AGIC Runtime 将数据分成几个主要 Zone：

|Zone|用途|
|---|---|
|`inst`|NPU 指令|
|`config`|图配置数据|
|`offline`|权重、常量|
|`custom`|算子输入和输出|
|`online`|Kernel 中间临时数据|
|`kvcache`|Attention KV Cache|

构造阶段通过 `set_state_dict_infos()` 读取每个文件大小，并进行内存对齐：

- `inst/config`：按 `0x1000` 对齐
- `offline`：按 `0x80` 对齐

之后 `set_zone_size()` 计算各 Zone 的总大小。

## 3. 子类初始化

具体算子继承 `Operator`，例如 `Add`：

```
Add::Add(string ckpt_dir, string op_name)
    : Operator(ckpt_dir, op_name)
{
    config_args();
    std::tie(m_out_size, m_out_dtype) = get_output_size(1);
    validate_custom_map();
}
```

主要工作包括：

### `config_args()`

根据 `agic_module.json` 中的参数槽位创建 Kernel 参数列表：

```
m_args.args[0]
m_args.args[1]
m_args.args[2]
...
```

参数类型分为：

```
ARG_INT      地址偏移、token_len、batch_size 等
ARG_POINTER  Host 指针参数
```

代码见 [operator.cpp (line 522)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:522)。

### `set_weight()`

有权重的算子会调用：

```
set_weight();
```

它把 offline Zone 中的权重偏移写入对应 Kernel 参数槽位：

```
m_args.args[slot].value.i = offline_offset;
```

这里绑定的是相对 offline Zone 的偏移，不是完整物理地址。

### `get_output_size()`

根据输出 shape、token 长度和 dtype 计算每个 Tile 的输出内存大小：

```
output_size =
    token_len
    × 其他维度
    × dtype_size
```

## 4. 加载指令、配置和权重

模型初始化时调用：

```
op->load_state_dict();
```

实现见 [operator.cpp (line 264)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:264)。

它会：

1. 创建或规划各个内存 Zone。
2. 读取 `inst/config/offline` 文件。
3. 将文件内容复制到对应 NPU 内存。
4. 记录每个 Zone 的起始偏移。

这个阶段通常每个 Operator 对象只执行一次。

## 5. 具体算子的 `forward()`

以 `Add` 为例，见 [add.cpp (line 31)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/add.cpp:31)。

### 5.1 分配输出

```
AgicTensor out_data =
    agicmem_manager->alloc_zone(
        m_out_size * input_1.token_len,
        "custom");
```

输入和输出通常都位于 `custom` Zone。

### 5.2 设置输出属性

```
out_data.token_len = input_1.token_len;
out_data.dtype = m_out_dtype;
set_shape(out_data);
```

`set_shape()` 根据编译产物中的 shape 描述设置逻辑 shape，并替换动态 token 维度。

### 5.3 绑定输入输出

```
set_custom({input_1, input_2, out_data});
```

顺序必须和 `custom_map`/`agic_module.json` 中的参数顺序一致。

`set_custom()` 取出每个 tensor 在 custom Zone 中的偏移：

```
offset = agicmem_manager->get_custom_offset(tensor.id);
```

然后写入 Kernel 参数：

```
m_args.args[slot].value.i = offset;
```

如果有 `token_len` 参数，也会自动绑定：

```
m_args.args[token_len_slot].value.i =
    tensors.begin()->token_len;
```

## 6. 执行 Kernel

具体算子调用：

```
Operator::forward();
```

它会根据是否启用 profiler，选择：

```
forward_common()
forward_profiler()
```

普通执行流程见 [operator.cpp (line 479)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:479)。

### 6.1 创建运行函数

```
set_args();
```

内部调用：

```
agFuncCreate(&m_func);
```

并为每个 Tile 设置相同的 Kernel 参数描述。

### 6.2 设置各 Zone 基地址

```
set_zone_offset();
```

生成：

```
m_offset_config.instAddr
m_offset_config.configAddr
m_offset_config.offlineAddr
m_offset_config.customAddr
m_offset_config.onlineAddr
m_offset_config.kvcacheAddr
```

可以把参数理解为：

```
实际地址 = Zone 基地址 + Kernel 参数中的相对偏移
```

然后把这些基地址写入 Kernel 配置：

```
memcpy(
    m_func->npu_pcfg,
    &m_offset_config,
    sizeof(agicInstConfig));
```

### 6.3 启动 Kernel

```
agicLaunchKernel(
    m_func,
    NULL,
    NULL,
    NULL,
    0,
    m_stream);
```

### 6.4 等待完成

```
agicStreamSynchronize(m_stream);
```

同步完成后，输出已经写入之前分配的 `out_data`。

## 7. 执行后处理

Kernel 完成后，一般会进行：

```
check_similarity(out_data);
dump_io_data({inputs...}, {out_data});
```

### `check_similarity()`

当：

```
CHECK_SIMILARITY=1
```

从 `GT_PATH` 加载标准 NPZ：

```
<op_name>.io.npz
```

找不到时再尝试：

```
<op_name>.output.npz
```

然后比较：

- Cosine Similarity
- MSE
- 部分 GT 和实际数据样本

### `dump_io_data()`

当：

```
DUMP_DATA=1
```

生成本次运行的：

```
<op_name>.input.npz
<op_name>.output.npz
```

它只负责 dump，不负责精度计算。

## 8. 资源释放

算子根据 `release` 参数决定是否释放输入：

```
if (release) {
    agicmem_manager->free_zone(input.id);
}
```

输出一般不会立即释放，因为要作为下一个算子的输入。

最终形成类似下面的内存流：

```
算子 A 输出 out_A
      ↓
算子 B 输入 out_A
      ↓
算子 B 执行完成
      ↓
释放 out_A
      ↓
返回 out_B
```

Operator 对象析构时还会执行：

```
agFuncDestroy(m_func);
```

## 9. Attention 的特殊逻辑

Attention 的主体执行流程相同，但会多出：

- 分配 `online` 临时空间
- 绑定 `batch_size`
- 绑定 prefill/decode `stage_id`
- 绑定 KV Cache 索引
- 设置 pointer 参数
- 绑定 kvcache Zone
- Kernel 前后 dump KV Cache
- 执行完成后释放 online Zone

大致流程：

```
生成 position/cache index
        ↓
分配输出 custom Zone
        ↓
分配 online Zone
        ↓
set_custom(输入、索引、输出)
        ↓
set_custom(token_len、batch_size、stage_id)
        ↓
set_online()
        ↓
Operator::forward()
        ↓
check_similarity / dump
        ↓
释放 online 和临时参数
```

Attention 示例见 [attentiononline.cpp (line 345)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/attentiononline.cpp:345)。

最核心的理解是：具体算子负责“准备什么数据”，`Operator` 基类负责“怎样把这些数据交给 Kernel 并运行”。
