
## 总共有几个区域

当前 `AgicMem_manager` 实际维护 7 个逻辑区域：

|区域|类型|用途|Kernel 是否直接使用|
|---|---|---|---|
|`inst`|静态|NPU 指令程序|是|
|`config`|静态|图和 Kernel 配置|是|
|`offline`|静态|权重、常量、tile_map|是|
|`event_log`|动态管理但固定大小|Profiler/Event 日志|否，单独传给日志接口|
|`kvcache`|动态|Attention KV Cache|是|
|`online`|动态临时|Kernel 中间 workspace|是|
|`custom`|动态|输入、输出、动态参数 Tensor|是|

因此：

- Kernel 的 `agicInstConfig` 中主要传 6 个区域：`inst/config/offline/kvcache/online/custom`。
- 内存管理器内部还有一个固定的 `event_log`。
- 总计 7 个逻辑区域。

定义位置：[agicmem_manager.cpp (line 139)](/localdata/env/sim_npu/share/yelinzhong/AGICInfer/agiccore/agic_memory/agicmem_manager.cpp:139)

## 总内存有多大

初始化设备时默认申请：

```
1024 * 1024 * (300 - 30)
```

也就是：

```
270 MiB
```

定义在 [agic_device.h (line 43)](/localdata/env/sim_npu/share/yelinzhong/AGICInfer/agiccore/agic_device/agic_device.h:43)。

也可以自行指定：

```
Init_AgicDevice(
    device_id,
    custom_memory_size);
```

内存管理器一次申请整块内存：

```
agicLocalMalloc(
    &m_base_address,
    -1,
    m_total_mem);
```

后面所有 Zone 都是这块大内存中的相对偏移。

## 内存排布

整体设计是静态区从低地址向高地址排列，custom 从最高地址向低地址增长：

```
低地址                                                       高地址
base                                                         base + total_mem
 │                                                                  │
 ▼                                                                  ▼
┌──────┬────────┬─────────┬───────────┬──────────────┬────────┬───────┐
│ inst │ config │ offline │ event_log │   kvcache    │ online │custom │
└──────┴────────┴─────────┴───────────┴──────────────┴────────┴───────┘
  固定     固定      固定      固定       向高地址增长    临时     向低地址增长
```

更准确地说，中间部分不是预先全部切死的：

```
┌─────────────┐ 低地址
│ inst        │
├─────────────┤
│ config      │
├─────────────┤
│ offline     │
├─────────────┤
│ event_log   │ 固定 1 MiB
├─────────────┤ ← kvcache.start
│             │
│ KV Cache    │ 从低地址方向使用
│             │
├─────────────┤ ← online.start
│ online      │ 每次 forward 临时申请
├─────────────┤ ← custom.start
│ custom      │ 输入/输出/参数
├─────────────┤
│ total_mem   │
└─────────────┘ 高地址
```

核心思想是：

```
kvcache  ↑ 向高地址使用

custom   ↓ 向低地址增长

online 放在二者中间，并且紧贴 custom
```

## 1. inst 区大小

`inst` 保存算子的 NPU 指令，例如：

```
test.bin
```

每个指令文件按 `0x1000`，即 4 KiB 对齐：

```
align_size = align(file_size, 0x1000);
```

所有指令文件的对齐大小相加后，总大小再做一次 4 KiB 对齐：

```
inst_size =
    align(sum(each_inst_align_size), 0x1000);
```

例如三个算子的 inst 文件：

```
op0 inst = 3000 bytes → 4096
op1 inst = 5000 bytes → 8192
op2 inst = 1000 bytes → 4096
```

那么：

```
inst zone = 4096 + 8192 + 4096
          = 16384 bytes
          = 16 KiB
```

## 2. config 区大小

`config` 保存图配置、Kernel 配置，例如：

```
graph_cfg.bin
```

计算方式和 `inst` 一样：

```
config_size =
    align(sum(each_config_align_size), 0x1000);
```

每个 config 文件按 4 KiB 对齐。

## 3. offline 区大小

`offline` 保存：

- 权重。
- 常量。
- tile_map。
- 不随每次 forward 改变的数据。

每个文件按 `0x80`，即 128 字节对齐：

```
align_size = align(file_size, 0x80);
```

总大小：

```
offline_size =
    sum(each_offline_file_aligned_to_128);
```

当前代码没有再把整个 offline zone 对齐到 4 KiB：

```
// offline_size = align(offline_size, 0x1000);
```

例如：

```
weight0 = 1000 bytes → 1024
weight1 = 2000 bytes → 2048
weight2 = 300 bytes  → 384
```

那么：

```
offline_size = 1024 + 2048 + 384
             = 3456 bytes
```

各权重的 offset 是：

```
weight0 offset = 0
weight1 offset = 1024
weight2 offset = 3072
```

这些偏移存入：

```
m_offline_offset_infos
```

再由 `set_weight()` 传给 Kernel。

## 4. event_log 区大小

固定为：

```
0x100000
```

即：

```
1 MiB
```

代码：

```
unsigned int m_event_log_size = 0x100000;
```

只用于 profiler/event log，不存普通 Tensor。

## 5. kvcache 区大小

初始化时：

```
m_zone_infos_dynamic["kvcache"] =
    Zone(m_start_offset, 0);
```

因此它的：

```
start   = event_log 结束位置
size    = 0
current = start
```

这里的 `size=0` 不表示不能使用 KV Cache，而是表示 KV Cache 没有像静态区那样提前切出固定容量。

它实际能够使用的最大范围是：

```
kvcache_limit =
    online.start - kvcache.start
```

如果没有 online，也可以理解为上界最终受 custom.start 限制：

```
可用 KV Cache 空间
≈ custom.start - kvcache.start
```

Attention 中的：

```
metadata.bind_kvcache_addr({0, ...});
```

传入的是相对于：

```
kvcache.start
```

的地址/偏移。

所以 KV Cache 是“有固定起点、没有预先固定终点”的区域。

## 6. custom 区大小

`custom` 存储：

- 输入 Tensor。
- 输出 Tensor。
- Attention 参数 Tensor。
- slot mapping。
- block table。
- request type。

custom 每次分配按：

```
0x100
```

即 256 字节对齐：

```
aligned_size =
    align(requested_size, 0x100);
```

第一次分配从总内存最高地址开始：

```
custom.start =
    total_mem - aligned_size;
```

例如总内存：

```
270 MiB
```

第一次申请 300 字节：

```
payload  = 300
capacity = 512
```

布局：

```
              total_mem
                  │
                  ▼
┌─────────────────┬─────────────┐
│ 剩余空闲区域    │ custom 512B │
└─────────────────┴─────────────┘
                  ↑
             custom.start
```

继续申请时，custom 通常向低地址增长：

```
                  高地址
┌─────────┬─────────┬─────────┐
│ output2 │ output1 │ input0  │
└─────────┴─────────┴─────────┘
↑
custom.start 向低地址移动
```

所以 custom 大小不是固定的：

```
custom.size =
    当前所有 slice 占用的 capacity 总和
```

释放中间 Tensor 后，可能产生空闲 slice，之后会尝试复用。

## 7. online 区大小

`online` 是每次 forward 的临时 workspace。

大小由 `Operator::get_online_size(token_len)` 动态计算：

```
online_size =
    各中间 Tensor 每 token 字节数
    × token_len
    - 可复用的内存
```

最终按照 256 字节对齐：

```
aligned_online_size =
    align(online_size, 0x100);
```

online 的位置固定贴着 custom 下边界：

```
online.start =
    custom.start - aligned_online_size;
```

也就是：

```
低地址                    高地址
┌─────────────┬──────────┬──────────┐
│ KV Cache    │ online   │ custom   │
└─────────────┴──────────┴──────────┘
              ↑          ↑
       online.start  custom.start
```

执行结束后：

```
release_online_workspace();
```

会将 online 重置为：

```
start = 0
size  = 0
```

所以在 forward 之外查询 online，通常会看到大小为 0。

## 每个区域大小总结

设：

```
T = total_mem，默认 270 MiB
```

那么：

|区域|大小计算|
|---|---|
|`inst`|所有 inst 文件分别 4 KiB 对齐后求和，总体再 4 KiB 对齐|
|`config`|所有 config 文件分别 4 KiB 对齐后求和，总体再 4 KiB 对齐|
|`offline`|所有权重文件分别 128 B 对齐后求和|
|`event_log`|固定 1 MiB|
|`kvcache`|不预先固定，最大到 online/custom 边界|
|`online`|每次 forward 计算，最后 256 B 对齐|
|`custom`|所有活跃 allocation 的对齐容量，单次分配按 256 B 对齐|

## 具体起始地址公式

设当前模型静态区起点为 `S`：

```
inst.start      = S
config.start    = S + inst_size
offline.start   = S + inst_size + config_size
event_log.start = S + inst_size + config_size + offline_size
kvcache.start   = event_log.start + 1 MiB

custom 第一次分配：
custom.start    = total_mem - custom_size

online：
online.start    = custom.start - online_size
```

因此必须满足：

```
kvcache.current
    <= online.start
    < custom.start
    <= total_mem
```

否则会抛出 `std::bad_alloc`。

## 如何查看运行时实际大小

在 `load_state_dict()` 之后，可以打印：

```
auto* manager =
    AgicDevice::getInstance().get_AgicMem_manager();

auto inst =
    manager->get_zone_addr("inst", "static");
auto config =
    manager->get_zone_addr("config", "static");
auto offline =
    manager->get_zone_addr("offline", "static");

auto event_log =
    manager->get_zone_addr("event_log");
auto kvcache =
    manager->get_zone_addr("kvcache");
auto online =
    manager->get_zone_addr("online");
auto custom =
    manager->get_zone_addr("custom");
```

每个 Zone 的有效范围一般查看：

```
start
size
current
use_addr
```

不过需要注意：

- `online.size` 只有 forward 期间非零。
- `custom.size` 随 Tensor 分配和释放变化。
- `kvcache.size` 当前通常保持 0，应主要查看 `kvcache.start` 和 `get_kvcache_limit()`。
- `inst/config/offline` 的大小取决于具体模型导出文件，无法只根据源码给出固定数值。