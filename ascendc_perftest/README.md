# ascendc_perftest

> **暂不支持 Ascend950**：当前暂不支持在 Ascend950 平台配套编译运行。

## 示例概述

ascendc_perftest是一个用于**测试AscendC::DataCopy性能并可与shmem的put/get接口进行比较**的参数化测试示例，支持多种测试类型、多种数据类型和灵活的参数配置。该示例可以测试不同核数和数据量下的性能表现，帮助用户评估和比较两种数据传输方式的性能差异，为性能优化提供参考。**该脚本测试结果仅做参考，性能以实际场景为准**

## 测试目的

本示例主要用于比较以下两种数据传输方式的性能：

1. **AscendC::DataCopy**：基于Ascend C的数据传输接口
2. **shmem的put/get接口**：共享内存的数据传输接口

通过对比测试结果，用户可以了解在不同场景下哪种数据传输方式具有更好的性能表现。

## 功能特性

### 支持的测试类型

- **put**: 测试跨设备PUT操作的性能（从本地设备内存传输到远端设备内存）
- **get**: 测试跨设备GET操作的性能（从远端设备内存传输到本地设备内存）
- **ub2gm_local**: 测试本地设备UB到GM的数据传输性能
- **ub2gm_remote**: 测试远端设备UB到GM的数据传输性能
- **gm2ub_local**: 测试本地设备GM到UB的数据传输性能
- **gm2ub_remote**: 测试远端设备GM到UB的数据传输性能
- **all**: 运行所有测试类型

### 支持的数据类型

- **浮点型**: `float`
- **有符号整型**: `int8`, `int16`, `int32`, `int64`
- **无符号整型**: `uint8`, `uint16`, `uint32`, `uint64`
- **字符型**: `char`
- **all**: 测试所有数据类型

### 其他特性

- ✅ 可配置核数范围（默认32核）
- ✅ 可配置数据量范围（支持2的幂次方）
- ✅ 可指定设备ID进行跨设备测试
- ✅ 可设置循环次数以获得稳定性能数据
- ✅ 自动数据正确性验证
- ✅ 生成CSV格式性能报告
- ✅ 支持批量测试（"all"选项）

## 编译说明

在shmem根目录下编译示例：

```bash
# 方法1：使用build.sh脚本编译所有示例
bash scripts/build.sh -examples

# 方法2：手动编译
mkdir -p build
cd build
cmake -DUSE_EXAMPLES=ON ..
source /usr/local/Ascend/ascend-toolkit/set_env.sh
make -j
```

## 使用方法

### 基本用法

```bash
# 进入示例目录
cd examples/shmem_perftest/ascendc_perftest

# 运行测试脚本
source /usr/local/Ascend/ascend-toolkit/set_env.sh
./run.sh [选项]
```

### 命令行参数

| 参数 | 缩写 | 描述 | 默认值 |
|------|------|------|--------|
| `--test-type <type>` | `-t <type>` | 测试类型 (put/get/ub2gm_local/ub2gm_remote/gm2ub_local/gm2ub_remote/all) | put |
| `--datatype <type>` | `-d <type>` | 数据类型 (float/int8/int16/int32/int64/uint8/uint16/uint32/uint64/char/all) | float |
| `--block-size <size>` | `-b <size>` | 设置单个核数 | 32 |
| `--block-range <min> <max>` | - | 设置连续核数范围 | 32-32 |
| `--block-list <b1,b2,...>` | - | 设置离散核数列表，如 `2,4,6,8` | - |
| `--exponent <exponent>` | `-e <exponent>` | 设置数据量的幂数 (2^exponent) | - |
| `--exponent-range <min> <max>` | - | 设置数据量的幂数范围 | 3-20 |
| `--device1 <id>` | - | 设置设备1 ID | 1 |
| `--device2 <id>` | - | 设置设备2 ID | 2 |
| `--loop-count <count>` | - | 设置循环次数 | 1000 |
| `--ub-size <size>` | - | 设置UB size(KB)，默认16 | 16 |

### 使用示例

#### 1. 测试单个操作类型和数据类型

```bash
# 测试PUT操作，float类型，32核，数据量从2^8到2^20字节，100次循环
./run.sh -t put -d float --block-size 32 --exponent-range 8 20 --device1 1 --device2 2 --loop-count 1000

# 测试GET操作，int32类型
./run.sh -t get -d int32 --block-size 32 --exponent-range 8 20 --device1 1 --device2 2 --loop-count 1000
```

#### 2. 测试所有操作类型（指定数据类型）

```bash
# 测试所有操作类型（PUT/GET/UB2GM/GM2UB），使用float数据类型
./run.sh -t all -d float --block-size 32 --exponent-range 8 20 --device1 1 --device2 2 --loop-count 1000
```

#### 3. 测试所有数据类型（指定操作类型）

```bash
# 测试PUT操作，使用所有数据类型
./run.sh -t put -d all --block-size 32 --exponent-range 8 20 --device1 1 --device2 2 --loop-count 1000
```

#### 4. 测试所有操作类型和数据类型

```bash
# 测试所有操作类型和所有数据类型
./run.sh -t all -d all --block-size 32 --exponent-range 8 20 --device1 1 --device2 2 --loop-count 1000
```

#### 5. 测试指定核数

```bash
# 在指定核数（2、4、6、8、16核）测试所有操作类型和所有数据类型
./run.sh -t all -d all --block-list 2,4,6,8,16 --exponent-range 8 20 --device1 1 --device2 2 --loop-count 1000
```

#### 6. 使用默认参数运行

```bash
# 使用默认参数运行put测试
./run.sh

# 运行所有测试类型
./run.sh -t all
```

#### 7. 图形化数据

```bash
# 使用perf_data_process.py -d 传入csv数据存储路径，图形化结果将存储在csv存储路径下picture文件夹下。
python ../utils/perf_data_process.py -d output
```

## 输出说明

### CSV文件格式

测试运行后，会在当前目录下生成`output`文件夹，其中包含测试结果的CSV文件，文件名格式为：

```bash
{rank}_{test_type}_{data_type}.csv
```

例如：

- `0_put_float.csv` - PUT操作，float类型
- `0_get_int32.csv` - GET操作，int32类型
- `0_ub2gm_local_float.csv` - 本地UB到GM操作，float类型
- `0_gm2ub_remote_int32.csv` - 远端GM到UB操作，int32类型

### CSV文件内容

CSV文件包含以下字段：

| 字段 | 描述 |
|------|------|
| `DataSize/B` | 数据大小（字节） |
| `Npus` | 使用的NPU数量 |
| `Blocks` | 核数 |
| `UBsize/KB` | UB大小（KB） |
| `Bandwidth/GB/s` | 带宽大小（GB/s） |
| `CoreMaxTime/us` | 最大核时间（微秒） |
| `SingleCoreTime/us` | 单核时间（微秒） |

## 数据验证

测试过程中仅PUT/GET会自动进行数据正确性验证：

- **PUT操作**: 验证源数据是否完整传输到目标地址
- **GET操作**: 验证从远端读取的数据是否正确

验证成功时会显示：

```bash
[Verification] SUCCESS: All cores' first values transferred correctly!
```

## 注意事项

1. **环境配置**：运行前请确保已正确设置Ascend环境变量：

   ```bash
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

2. **设备配置**：确保指定的设备ID（device1和device2）可用且支持Peer Access

3. **数据量计算**：测试数据量的计算方式为：`2^exponent` 字节

4. **参数范围**：核数范围不能大于设备最大值

5. **内存要求**：大数据量测试需要足够的设备内存

6. **数据类型支持**：某些数据类型（如uint64、char）可能在特定场景下存在限制

7. **性能稳定性**：建议使用较大的循环次数（如1000次）以获得稳定的性能数据

## 性能分析建议

1. 对比不同测试类型（put/get/ub2gm_local/ub2gm_remote/gm2ub_local/gm2ub_remote）的性能差异
2. 对比不同数据类型的性能表现
3. 根据实际硬件配置调整核数参数，观察性能变化
4. 针对不同的数据量范围分析性能瓶颈
5. 结合测试结果选择最适合实际应用场景的数据传输方式和数据类型
6. 对比本地操作（_local）和远端操作（_remote）的性能差异

## 故障排查

### 常见问题

1. **设备不可用错误**

   ```bash
   错误: 设备ID无效
   ```

   解决方案：使用 `npu-smi info` 检查可用设备，确保device1和device2参数正确

2. **内存分配失败**

   ```bash
   错误: aclrtMalloc failed
   ```

   解决方案：减小数据量范围或核数，确保设备有足够内存

3. **Peer Access失败**

   ```bash
   错误: aclrtDeviceEnablePeerAccess failed
   ```

   解决方案：确保两个设备之间支持Peer Access

4. **数据验证失败**

   ```bash
   [Verification] FAILED: Data mismatch
   ```

   解决方案：检查设备状态，可能需要重置设备或重新运行测试。
