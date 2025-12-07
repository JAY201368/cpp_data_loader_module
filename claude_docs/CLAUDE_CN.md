# CLAUDE.md

本文档为Claude Code（claude.ai/code）在此代码库中处理代码时提供指导。

## 项目概述

这是一个用于NNUE（高效可更新神经网络）国际象棋引擎训练的C++训练数据加载器。它从`.bin`和`.binpack`文件中加载国际象棋局面数据，根据可配置的特征集提取稀疏特征，并通过ctypes绑定为基于Python的神经网络训练提供批处理数据。

该数据加载器对性能要求极高，专为多线程操作设计，以避免成为GPU/CPU训练流程的瓶颈。

## 构建命令

### 标准构建（RelWithDebInfo）
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
make -j
```

或使用提供的脚本：
```bash
sh compile_data_loader.bat
```

### 构建类型
- `RelWithDebInfo`：带调试符号的发布版本（默认，推荐用于本地开发）
- `Release`：完全优化版本（`-O3 -march=native -DNDEBUG`）
- `Debug`：带符号的调试版本（`-g`）

### 配置文件引导优化（PGO）构建
PGO构建需要两步流程：

1. **生成性能分析构建：**
```bash
cmake -S . -B build-pgo-generate -DCMAKE_BUILD_TYPE=PGO_Generate
cmake --build ./build-pgo-generate --config PGO_Generate
```

2. **运行基准测试以收集性能分析数据：**
```bash
./build-pgo-generate/training_data_loader_benchmark 路径/到/data.binpack
```

3. **使用性能分析数据构建优化二进制文件：**
```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=PGO_Use \
    -DPGO_PROFILE_DATA_DIR=build-pgo-generate/pgo_data \
    -DCMAKE_INSTALL_PREFIX="./"
cmake --build ./build --config PGO_Use --target install
```

4. **清理：**
```bash
rm -rf build-pgo-generate
```

### 基准测试构建
构建基准测试可执行文件：
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=PGO_Generate
cmake --build ./build --config PGO_Generate
```

运行：
```bash
./build/training_data_loader_benchmark 路径/到/data.binpack
```

## 架构

### 核心组件

**training_data_loader.cpp**：暴露C API供Python绑定的主入口点
- 实现多个国际象棋局面特征集（HalfKP、HalfKA、HalfKAv2、HalfKAv2_hm和自定义HalfATA）
- 特征集可"因子化"（用`^`后缀标记）以分离真实特征和虚拟特征
- 提供`SparseBatch`和`FenBatch`输出格式
- 使用生产者-消费者模式和工作线程进行并行数据处理

**lib/nnue_training_data_formats.h**：国际象棋局面表示和二进制格式处理
- 实现`Position`、`Move`、`Piece`、`Square`类型
- 处理压缩binpack格式的读取/写入
- 提供局面操作（走子生成、FEN解析）
- 约270KB文件，包含完整的国际象棋逻辑实现

**lib/nnue_training_data_stream.h**：文件I/O和流基础设施
- `BinSfenInputStream`：读取`.bin`格式
- `BinpackSfenInputStream`：读取`.binpack`压缩格式
- `BinpackSfenInputParallelStream`：多线程binpack读取器
- 支持循环读取（在数据集上循环）和跳过谓词

**lib/rng.h**：用于多线程过滤的线程局部随机数生成

### 特征集

代码库实现了多种NNUE特征表示：

1. **HalfKP**：王-棋子特征（41,024个输入）
    - 王位置（64）×（棋子类型（10）×位置（64）+ 1）

2. **HalfKA**：王-所有棋子包括对手的王（49,216个输入）
    - 王位置（64）×（棋子类型（12）×位置（64）+ 1）

3. **HalfKAv2**：紧凑的王-所有编码（45,056个输入）
    - 将对手的王打包到相同的特征空间中

4. **HalfKAv2_hm**：水平镜像变体（22,528个输入）
    - 使用王分区将特征空间减少50%
    - 王的E-H线被镜像到D-A线

5. **HalfATA**（自定义，未完成）：基于攻击的特征
    - 第370-402行、431-458行的TODO项
    - 按攻击分区而不是王位置对局面进行分类

每个特征集都有一个"因子化"变体（后缀`^`），添加了虚拟特征，将王位置与棋子位置分离以改进训练。

### 数据流

1. **文件读取线程**：从`.bin`/`.binpack`文件读取原始训练条目
2. **特征提取工作线程**：将国际象棋局面转换为稀疏特征向量
3. **批次组装**：将条目分组到`SparseBatch`中，具有对齐的数组以便高效GPU传输
4. **Python消费**：`nnue_dataset.py`中的ctypes绑定（不在此代码库中）消费批次

线程分配：`concurrency`参数在读取线程和特征线程之间按1:2分配。

### 关键数据结构

**TrainingDataEntry**：单个训练局面
- `Position pos`：国际象棋棋盘状态
- `Move move`：最佳走子
- `int16_t score`：引擎评估
- `int16_t ply`：走子编号
- `float result`：游戏结果（-1/0/+1）

**SparseBatch**：用于神经网络的训练数据批次
- `int* white/black`：每方的稀疏特征索引
- `float* white_values/black_values`：稀疏特征值（主要为1.0）
- `float* score/outcome/is_white`：训练标签
- `int* layer_stack_indices`：基于棋子数量的分区索引

**局面方向**：
- 白方视角：棋盘位置不变
- 黑方视角：`flippedVertically().flippedHorizontally()`（180°旋转）
    - 注意：第51-52行的注释表明这是为了Stockfish兼容性

## 自定义特征集开发

要添加新的特征集（示例：HalfATA已部分实现）：

1. 定义特征集结构体，包含：
    - `static constexpr int INPUTS`：总特征维度
    - `static constexpr int MAX_ACTIVE_FEATURES`：每个位置的最大激活特征数
    - `static int feature_index(...)`：将局面组件映射到特征索引
    - `static std::pair<int, int> fill_features_sparse(...)`：提取激活特征

2. 如果需要，添加因子化变体（参见第431-459行的`HalfATAFactorized`）

3. 在C API函数中注册：
    - `get_sparse_batch_from_fens`（第1121行）
    - `create_sparse_batch_stream`（第1204行）

4. 实现`classify_attack_buckets()`或类似的局面分类器（第399行存根）

5. 更新Python端的特征集配置

## 平台特定说明

**BMI2支持**：CMake自动检测BMI2 CPU指令支持
- 通过`_pdep_u64`内置函数实现更快的位棋盘操作
- 检测逻辑位于CMakeLists.txt:24-60
- 仅对非AMD CPU或AMD Zen 2+（系列≥23）启用

**导出宏**：
- x86_64：无需特殊导出
- Windows MSVC：`__declspec(dllexport)`和`__cdecl`
- 其他：默认导出

## 代码注释

此代码库在training_data_loader.cpp中包含大量中文注释，解释：
- 数据加载器架构和设计决策
- 因子化特征的特征提取逻辑
- 多线程模式（生产者-消费者）
- SparseBatch内存布局

中文注释是有意为之的文档，在修改代码时应予以保留。

## 依赖项

- 需要C++17（在CMakeLists.txt:19-20中设置）
- CMake 3.10+
- 线程库（Unix上的pthread）
- 无需外部库（国际象棋逻辑自包含）

## 输出

**共享库**：`libtraining_data_loader.so`（Linux/macOS）或`training_data_loader.dll`（Windows）
- 通过`make install`安装到CMAKE_INSTALL_PREFIX
- 通过ctypes由Python训练脚本加载

**基准测试可执行文件**：`training_data_loader_benchmark`
- 仅在`CMAKE_BUILD_TYPE=PGO_Generate`时构建
- 测量吞吐量（MPos/s、iterations/s、MB/s、bytes/position）