核心结论：`Operator` 是所有算子 Wrapper 的执行基类。派生算子负责“分配输入输出、组织参数”，`Operator` 负责“解析 metadata、加载二进制、计算内存地址、启动 kernel、校验与 dump”。

源码入口：

- [operator.h (line 13)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.h:13)
- [operator.cpp (line 20)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:20)

## 一次算子的典型调用流程

```
构造派生算子
  ├─ Operator(...)
  ├─ config_args()
  ├─ set_weight()
  ├─ get_output_size()
  └─ validate_custom_map()

加载模型
  └─ load_state_dict()
       ├─ set_zone()
       └─ 加载 inst/config/offline

执行算子
  ├─ 分配输出 Tensor
  ├─ set_custom()
  ├─ set_pointer_args()
  ├─ set_online()
  └─ Operator::forward()
       ├─ forward_common()
       └─ forward_profiler()

执行之后
  ├─ check_similarity()
  ├─ dump_io_data()
  └─ 释放输入/临时内存
```

注意：这里存在两层 `forward`。例如 `Rmsnorm::forward_common()` 负责算子业务包装，而 `Operator::forward()` 只负责真正启动 kernel。

---

## 一、构造、析构与模型加载

### 1. `Operator(string ckpt_dir, string op_name="")`

实现：[operator.cpp (line 30)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:30)

作用：初始化一个算子的所有通用 metadata。

主要工作：

- 保存算子目录和算子名。
- 读取 `agic_module.json`。
- 解析：

```
custom 输入输出顺序
online 临时 Tensor
offline 权重
runtime var 参数槽位
extra 参数
输入输出 shape/dtype
inst/config/offline 文件
```

- 检测是否为新的 cache 模式。
- cache 模式下重映射共享指令和分层权重路径。
- 调用 `set_state_dict_infos()` 整理待加载文件。
- 调用 `set_zone_size()` 统计静态内存需求。
- 获取 tile 数量、stream、profiler 配置。
- 根据 `PROFILER_EVA` 选择 `forward_common` 或 `forward_profiler`。

前提：构造 `Operator` 前必须已经调用 `Init_AgicDevice()`，否则 `agicmem_manager` 可能为空。

---

### 2. `~Operator()`

实现：[operator.cpp (line 181)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:181)

作用：销毁当前保存的 runtime function：

```
agFuncDestroy(m_func);
```

注意：

- 当前析构函数不是 `virtual`。
- 不建议通过 `Operator*` 删除派生类对象。
- `set_args()` 每次执行都会调用 `agFuncCreate()`，但析构只销毁当前的 `m_func`，阅读生命周期时要特别留意。

---

### 3. `set_state_dict_infos(load_infos, zone_name)`，private

实现：[operator.cpp (line 186)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:186)

作用：把 JSON 中解析出的 `StZoneInfo` 转成真正可加载的 `StDataInfo`。

具体工作：

- 将文件名中的 `*` 替换为 tile 0，用于获取文件大小。
- 区分绝对路径和相对路径。
- 获取二进制文件大小。
- 判断是否为 per-tile 数据。
- 处理 merged-tile 权重。
- 计算内存对齐大小：

```
offline：0x80 对齐
inst/config：0x1000 对齐
```

- 保存到 `m_state_dict_infos[zone_name]`。

它目前主要处理：

```
inst
config
offline
```

---

### 4. `set_zone_size()`，private

实现：[operator.cpp (line 227)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:227)

作用：统计该算子需要的静态 Zone 大小。

具体工作：

- 累加所有 inst 文件大小。
- 累加所有 config 文件大小。
- 累加所有 offline 权重大小。
- 为每个 offline 权重计算相对于 offline zone 的偏移。
- 调用：

```
agicmem_manager->set_zone_size(
    inst_size,
    cfg_size,
    offline_size
);
```

`set_weight()` 后面绑定的就是这里算出来的 offline 相对偏移。

---

### 5. `load_state_dict()`

实现：[operator.cpp (line 264)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:264)

作用：真正把算子的指令、配置和权重二进制加载到设备内存。

执行过程：

```
agicmem_manager->set_zone()
→ 遍历 inst/config/offline
→ set_zone_data()
→ 记录当前算子在各静态 zone 中的起始位置
```

`m_zone_offsets` 保存的是当前算子自己的：

```
inst 起始偏移
config 起始偏移
offline 起始偏移
```

注意：这里的 `state_dict` 不只是神经网络权重，也包含 kernel instruction 和 config。

---

## 二、Kernel 参数结构初始化

### 6. `config_args(pointer_indexs={})`

实现：[operator.cpp (line 522)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:522)

作用：根据 `agic_module.json` 的 `var` 槽位初始化 kernel 参数数组。

具体工作：

- 找出最大的参数槽位编号。
- 设置 `m_args.arg_count`。
- 创建对应数量的 `arg_t`。
- 默认参数类型为 `ARG_INT`。
- `pointer_indexs` 指定的位置初始化成 `ARG_POINTER`。

例子：

```
config_args();             // 普通算子
config_args({3, 4});       // 第 3、4 个槽位为 pointer
```

注意：它不会清空已有的 `m_args.args`，因此通常只应在构造函数里调用一次。

---

### 7. `set_args()`，private

实现：[operator.cpp (line 289)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:289)

作用：把已经准备好的 `m_args` 装入 AGIC runtime function。

具体工作：

- 调用 `agFuncCreate(&m_func)`。
- 设置参与执行的 tile 数量。
- 为每个 tile 设置同一份：

```
arg_count
args 指针
```

这一步只构造 runtime function，还没有启动 kernel。

---

### 8. `set_weight()`

实现：[operator.cpp (line 561)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:561)

作用：把每个 offline 权重的相对偏移写进 kernel 参数。

概念上是：

```
权重名
→ m_var_infos 中的参数槽位
→ offline zone 中的相对偏移
```

写入：

```
m_args.args[slot].value.i = offline_offset;
```

这里绑定的不是绝对设备地址，而是相对于 `offlineAddr` 的偏移。

---

## 三、Custom 输入输出参数绑定

### 9. `custom_tensor_slots() const`，private

实现：[operator.cpp (line 603)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:603)

作用：返回 `set_custom()` 应接收多少个 Tensor。

如果 `m_custom_offsets` 最后一项是 `token_len`，它不是 Tensor，因此数量减一：

```
[input, output, token_len]
→ Tensor 槽位数量为 2
```

---

### 10. `validate_custom_map() const`，protected

实现：[operator.cpp (line 583)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:583)

作用：检查 custom 参数约定是否基本正确。

检查内容：

- custom 参数列表不能是空的。
- 如果存在 `token_len`，它必须位于 custom 列表末尾。
- custom 名称是否能在 `m_var_infos` 中找到。

缺失变量目前通常只打印警告，因为 cache 格式中可能出现变量重命名。

---

### 11. `set_custom(initializer_list<AgicTensor>)`

实现：[operator.cpp (line 611)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:611)

作用：按照 metadata 中规定的顺序绑定输入输出 Tensor。

示例：

```
set_custom({input, output});
```

执行逻辑：

```
第 i 个 Tensor
→ 第 i 个 custom 名称
→ 对应 var 槽位
→ Tensor 在 custom zone 中的偏移
```

同时，如果 kernel 有 `token_len` 参数，会从第一个 Tensor 取：

```
tensors.begin()->token_len
```

关键要求：Tensor 顺序必须严格匹配 metadata 顺序。数量不一致会直接退出。

---

### 12. `set_custom({{name, AgicTensor}, ...})`

实现：[operator.cpp (line 649)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:649)

作用：根据参数名称绑定 Tensor，不依赖列表顺序。

示例：

```
set_custom({
    {"input", input_tensor},
    {"output", output_tensor},
});
```

它会把 Tensor 的 custom zone 偏移写入对应参数槽位。

适合：

- 参数顺序不方便维护。
- 只更新部分 Tensor 参数。
- Attention 等复杂算子。

注意：函数没有验证名称是否存在；错误名称可能通过 `map::operator[]` 意外落到槽位 0。

---

### 13. `set_custom({{name, int}, ...})`

实现：[operator.cpp (line 665)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:665)

作用：按名称设置普通整数/scalar 参数。

示例：

```
set_custom({
    {"token_len", 32},
    {"cache_blocks", 128},
    {"request_nums", 4},
});
```

常用于：

- `token_len`
- stage id
- batch size
- cache 数量
- cache 地址或偏移
- request 数量

同样要求 `name` 必须存在于 `m_var_infos`。

---

## 四、Pointer 参数

### 14. `set_pointer_args(init_list)`

实现：[operator.cpp (line 701)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:701)

作用：保存并绑定一组 host pointer 参数。

示例：

```
set_pointer_args({
    {"query_lens", query_lens},
    {"kv_lens", kv_lens},
});
```

它会先把 vector 复制到 `m_pointer_data`，然后绑定：

```
pointer = vector.data()
length  = vector.size() * sizeof(uint32_t)
```

复制到成员变量很重要，因为 runtime 使用 pointer 时，底层数据必须仍然存活。

前提：对应槽位必须在 `config_args(pointer_indexs)` 中声明为 `ARG_POINTER`。

---

### 15. `set_pointer_args()`

实现：[operator.cpp (line 680)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:680)

作用：把当前 `m_pointer_data` 中的数据重新绑定到参数槽位。

适用于派生类先修改：

```
m_pointer_data["query_lens"] = query_lens;
```

然后调用：

```
Operator::set_pointer_args();
```

`Qwen2Block::generate()` 使用的就是这种方式。

---

## 五、Online 临时内存

### 16. `set_online_info(reuse_infos={})`，protected

实现：[operator.cpp (line 319)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:319)

作用：根据 online metadata 建立每个 online Tensor 的描述。

保存的信息包括：

- shape
- dtype
- 动态 token_len 所在维度
- 每个 token 占用的字节数
- 是否复用其他 online buffer
- 被复用的 Tensor 名称

复用示例：

```
{
    {"query_states", "fused_qkv_states"},
    {"key_states", "fused_qkv_states"},
    {"value_states", "fused_qkv_states"},
}
```

---

### 17. `get_online_size(int token_len)`，protected

实现：[operator.cpp (line 351)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:351)

作用：

- 计算当前 token 长度下整个 online zone 需要多少字节。
- 同时计算每个 online Tensor 的 offset。
- 处理 online buffer 复用。

典型使用：

```
agicmem_manager->alloc_zone(
    get_online_size(input.token_len),
    "online"
);
```

因此它不只是查询函数，也会更新 `m_online_infos[name].offset`。

---

### 18. `set_online()`

实现：[operator.cpp (line 723)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:723)

作用：把已经算好的 online offset 写入 kernel 参数。

对于只用于 layout/dump、并不存在于 runtime `var` 中的 FakeSplit Tensor，会跳过绑定，防止误写参数槽位 0。

正确调用顺序通常是：

```
set_online_info()        构造时
get_online_size()        forward 时
alloc online zone
set_online()
```

---

## 六、输出大小和 Shape

### 19. `get_output_size(token_len, index=0)`，protected

实现：[operator.cpp (line 745)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:745)

作用：根据 output metadata 计算某个输出每个 tile 的字节数，并返回 dtype。

返回值：

```
{output_size_in_bytes, dtype}
```

当前计算约定：

```
token_len
× metadata shape[1...]
× dtype 字节数
```

常见用法：

```
std::tie(m_out_size, m_out_dtype) = get_output_size(1);
```

forward 时再分配：

```
m_out_size * input.token_len
```

`index` 用于多输出算子。

---

### 20. `output_has_dynamic_token_len(index=0) const`，protected

实现：[operator.cpp (line 759)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:759)

作用：判断指定输出 shape 中是否存在动态 `token_len` 维度。

判断条件：

```
token_len_index >= 0
```

主要用于区分：

- 动态文本输出：大小随 token_len 变化。
- 固定 Shape 输出：例如部分 Vision block。

---

### 21. `set_shape(AgicTensor&, index=0)`，protected

实现：[operator.cpp (line 764)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:764)

作用：完善输出 `AgicTensor` 的逻辑属性。

具体工作：

- 从 metadata 复制 shape。
- 将动态维度替换为 `tensor.token_len`。
- 更新 `m_token_len`，供 profiler 使用。
- 标记 Tensor 位于 NPU。
- 计算并设置 `tensor.npu_addr`。

通常在输出分配后调用：

```
out_data.token_len = input.token_len;
out_data.dtype = m_out_dtype;
set_shape(out_data);
```

---

### 22. `get_cache_dtype()`，protected

实现：[operator.cpp (line 1788)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:1788)

作用：从 custom 输出 metadata 中推断 KV cache dtype。

查找输出名：

```
past_key
past_kv
```

找到后返回对应 dtype；找不到时默认：

```
bfloat16
```

主要由 Attention 类算子使用。

---

## 七、真正启动 Kernel

### 23. `set_zone_offset()`，private

实现：[operator.cpp (line 302)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:302)

作用：计算每种 zone 的绝对设备地址，并写入 `agicInstConfig`。

设置：

```
instAddr
configAddr
offlineAddr
customAddr
onlineAddr
kvcacheAddr
```

地址公式大体为：

```
静态 zone：
device_base + 当前算子在 zone 中的偏移

动态 zone：
device_base + zone.start
```

kernel 参数内部保存的是各 Tensor/权重相对于这些 base address 的偏移。

---

### 24. `forward()`

实现：[operator.cpp (line 517)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:517)

作用：根据 profiler 配置分发到底层启动方式：

```
(this->*m_forward_fn)();
```

可能调用：

```
forward_common()
forward_profiler()
```

它不分配输入输出，也不绑定参数；这些工作必须由派生算子提前完成。

---

### 25. `forward_common()`

实现：[operator.cpp (line 479)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:479)

作用：普通模式下启动 kernel。

执行过程：

```
set_args()
→ set_zone_offset()
→ 将 agicInstConfig 复制给 runtime function
→ agicLaunchKernel()
→ agicStreamSynchronize()
```

如果开启了 `profile_run`，函数不会真正启动 kernel，而是更新 KV cache 内存上限后返回。这更像内存规划/试运行模式。

---

### 26. `forward_profiler()`

实现：[operator.cpp (line 392)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:392)

作用：带性能统计地启动 kernel。

相比 `forward_common()`，额外完成：

- 向 profiler 记录输出 shape。
- 开启 profiler event range。
- 初始化 event log。
- 记录 host 侧执行时间。
- 读取每个 tile 的事件和 cycle。
- 将性能数据写入 `AgicDeviceProfiler`。

仍然会在 kernel 后同步 stream。

---

## 八、精度验证

### 27. `check_similarity(result, index=0)`

实现：[operator.cpp (line 1540)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:1540)

作用：当前主要使用的 NPZ 精度验证接口。

只有启用：

```
export CHECK_SIMILARITY=1
```

才会执行。

流程：

- 从 `GT_PATH` 查找：

```
<op_name>.io.npz
<op_name>.output.npz
```

- `GT_KEY` 可以覆盖 GT key。
- 将 NPU 输出逐 tile 拷贝到 host。
- 根据 dtype 转成 float。
- 必要时应用 `valid_len` 过滤。
- 计算并打印：

```
Cosine Similarity
MSE
部分 GT/实际值样本
```

`index` 用于多输出算子。

---

### 28. `check_result(result, index, tile_num, zone_name, mode, save)`

实现：[operator.cpp (line 1630)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:1630)

作用：使用每个 tile 独立的 `.bin` 文件进行精度验证。

GT 路径自动生成：

```
<ckpt_dir>/<output_name>_tile_*.bin
```

它会：

- 根据 metadata 计算每个 tile 的输出大小。
- 读取所有 tile GT。
- 从设备读取所有 tile 输出。
- 计算 cosine similarity 和 MSE。
- `save=true` 时保存：

```
output_re.bin
output_gt.bin
```

`mode` 参数当前没有实际作用。

---

### 29. `check_result(result, tile_num, zone_name, bin_path, size, save)`

实现：[operator.cpp (line 1689)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:1689)

作用：使用调用方指定的 GT 路径和每 tile 字节数进行比较。

相比上一个重载，调用方明确提供：

- `bin_path`
- `size`
- `tile_num`
- `zone_name`

适合特殊算子或 metadata 无法自动推导 GT 路径的情况。

---

### 30. `check_result_offset(offset, tile_num, zone_name, bin_path, size, save)`

实现：[operator.cpp (line 1746)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:1746)

作用：直接根据 zone 内偏移读取结果并与 bin GT 比较。

它不需要 `AgicTensor`，调用者直接给：

```
zone 名称
zone 内 offset
每 tile 字节数
GT 路径
```

代码注释将其标记为临时代码，主要用于检查 online zone 中的中间数据。

---

## 九、普通输入输出 Dump

### 31. `dump_io_data(inputs, outputs)`

实现：[operator.cpp (line 1604)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:1604)

作用：统一 dump 一个算子的输入和输出。

由环境变量控制：

```
export DUMP_DATA=1
```

生成：

```
<op_name>.input.npz
  input_0
  input_1
  ...

<op_name>.output.npz
  output_0
  output_1
  ...
```

内部逐个调用 `dump_data()`。

---

### 32. `dump_data(result, path, data_name, index, data_info)`

实现：[operator.cpp (line 1068)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:1068)

作用：将单个 custom Tensor 从设备复制到 host 并写入 NPZ。

参数含义：

- `result`：待 dump 的 Tensor。
- `strPath`：NPZ 文件路径。
- `data_name`：NPZ 数组名称。
- `index`：对应 metadata 中的第几个输入/输出。
- `data_info`：通常是 `"input"` 或 `"custom"`。

它会从 metadata 补全 dtype、shape，并支持多种数据类型。

注意：虽然 `data_info` 可以是不同 metadata 分区，但读取地址固定来自 custom zone，因此主要用于普通 custom 输入输出。

---

## 十、Attention/KV Cache Dump 辅助函数

### 33. `online_map_has_fused_qkv() const`，protected

实现：[operator.cpp (line 20)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:20)

作用：判断 online metadata 中是否存在融合 QKV buffer。

识别名称：

```
fused_qkv_states
fused_qkv
```

Attention 派生类据此决定 query/key/value 是否复用同一块 online 内存。

---

### 34. `kvcache_byte_offset_from_addr(int addr) const`，protected

实现：[operator.cpp (line 791)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:791)

作用：把一个 KV cache 地址形式的值标准化成相对于 KV cache zone 的字节偏移。

它兼容两种输入：

```
已经是 offset
绝对设备地址
```

绝对地址会执行：

```
addr - kvcache_zone_base
```

主要供 KV cache dump 使用。

---

### 35. `dump_kvcache_slices_from_past(...) const`，protected

实现：[operator.cpp (line 917)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:917)

作用：根据 `past_k/past_v` 的 block 地址列表，dump 指定 KV cache block。

参数：

- `cache`：二维 block 地址/偏移列表。
- `block_bytes`：每个 block 字节数。
- `strPath`：NPZ 路径。
- `data_name`：数组名。
- `elem_dtype`：保存的数据类型。
- `npz_mode`：`"w"` 或 `"a"`。

它会按 `cache[b][i]` 顺序读取各 block，并将其拼接后按 tile 保存。

---

### 36. `dump_zone_linear_tiled(...) const`，protected

实现：[operator.cpp (line 976)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:976)

作用：通用的“从某个 zone 连续读取一段内存并按 tile 保存到 NPZ”函数。

参数指定：

```
zone_name
zone 内字节偏移
每 tile 字节数
dtype
可选逻辑 shape
```

这是后面 online/KV cache dump 函数的底层通用实现。

---

### 37. `dump_online_labeled_slices_to_npz(...) const`，protected

实现：[operator.cpp (line 1009)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:1009)

作用：把所有 online Tensor 分别 dump 到一个 NPZ。

数组名格式：

```
online_blk_<tensor_name>_<name_infix>
```

例如：

```
online_blk_query_states_before
online_blk_query_states_after
```

它会利用 `m_online_infos` 中保存的 offset、size、dtype 和 shape。

---

### 38. `dump_online_labeled_one_block_to_npz(...) const`，protected

实现：[operator.cpp (line 1042)](/localdata/env/sim_npu/share/yelinzhong/AGICExecutor/src/agic_operator/common/operator.cpp:1042)

作用：只 dump 一个指定的 online Tensor。

参数：

- `map_key`：online metadata 中的名称。
- `npz_data_name`：最终 NPZ 数组名。

适用于将某个 online 中间结果映射成标准输出名，例如 `output_1`。

---

## 最值得优先掌握的函数

第一遍不需要把所有 dump 函数都研究透。建议按这个优先级学习：

1. `Operator()`
2. `load_state_dict()`
3. `config_args()`
4. `set_weight()`
5. `set_custom()`
6. `get_output_size()`
7. `set_shape()`
8. `set_zone_offset()`
9. `forward_common()`
10. `check_similarity()`
11. `set_online_info()` / `get_online_size()` / `set_online()`
12. Pointer、KV cache 和 dump 辅助函数

掌握前十个，就能完整读懂 `Rmsnorm`、`Add`、`Gemm`、`Mlp` 这类普通算子。