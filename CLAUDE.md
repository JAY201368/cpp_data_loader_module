# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a C++ training data loader for NNUE (Efficiently Updatable Neural Network) chess engine training. It loads chess position data from `.bin` and `.binpack` files, extracts sparse features based on configurable feature sets, and provides batched data for Python-based neural network training via ctypes bindings.

The data loader is performance-critical and designed for multi-threaded operation to avoid bottlenecking GPU/CPU training pipelines.

## Build Commands

### Standard Build (RelWithDebInfo)
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
make -j
```

Or use the provided script:
```bash
sh compile_data_loader.bat
```

### Build Types
- `RelWithDebInfo`: Release with debug symbols (default, recommended for local development)
- `Release`: Full optimization (`-O3 -march=native -DNDEBUG`)
- `Debug`: Debug build with symbols (`-g`)

### Profile-Guided Optimization (PGO) Build
PGO builds require a two-step process:

1. **Generate profiling build:**
```bash
cmake -S . -B build-pgo-generate -DCMAKE_BUILD_TYPE=PGO_Generate
cmake --build ./build-pgo-generate --config PGO_Generate
```

2. **Run benchmark to collect profile data:**
```bash
./build-pgo-generate/training_data_loader_benchmark path/to/data.binpack
```

3. **Build optimized binary using profile data:**
```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=PGO_Use \
    -DPGO_PROFILE_DATA_DIR=build-pgo-generate/pgo_data \
    -DCMAKE_INSTALL_PREFIX="./"
cmake --build ./build --config PGO_Use --target install
```

4. **Cleanup:**
```bash
rm -rf build-pgo-generate
```

### Benchmark Build
To build the benchmark executable:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=PGO_Generate
cmake --build ./build --config PGO_Generate
```

Run with:
```bash
./build/training_data_loader_benchmark path/to/data.binpack
```

## Architecture

### Core Components

**training_data_loader.cpp**: Main entry point exposing C API for Python bindings
- Implements multiple chess position feature sets (HalfKP, HalfKA, HalfKAv2, HalfKAv2_hm, and custom HalfATA)
- Feature sets can be "factorized" (marked with `^` suffix) to separate real and virtual features
- Provides `SparseBatch` and `FenBatch` output formats
- Uses producer-consumer pattern with worker threads for parallel data processing

**lib/nnue_training_data_formats.h**: Chess position representation and binary format handling
- Implements `Position`, `Move`, `Piece`, `Square` types
- Handles compressed binpack format reading/writing
- Provides position manipulation (move making, FEN parsing)
- ~270KB file with complete chess logic implementation

**lib/nnue_training_data_stream.h**: File I/O and streaming infrastructure
- `BinSfenInputStream`: Reads `.bin` format
- `BinpackSfenInputStream`: Reads `.binpack` compressed format
- `BinpackSfenInputParallelStream`: Multi-threaded binpack reader
- Supports cyclic reading (looping over dataset) and skip predicates

**lib/rng.h**: Thread-local random number generation for multi-threaded filtering

### Feature Sets

The codebase implements multiple NNUE feature representations:

1. **HalfKP**: King-piece features (41,024 inputs)
   - King square (64) × (piece type (10) × square (64) + 1)

2. **HalfKA**: King-all pieces including opponent king (49,216 inputs)
   - King square (64) × (piece type (12) × square (64) + 1)

3. **HalfKAv2**: Compact king-all encoding (45,056 inputs)
   - Packs opponent king into same feature space

4. **HalfKAv2_hm**: Horizontal mirroring variant (22,528 inputs)
   - Uses king buckets to reduce feature space by 50%
   - King files E-H are mirrored to files D-A

5. **HalfATA** (Custom, incomplete): Attack-based features
   - TODO items at lines 370-402, 431-458
   - Classify positions by attack bucket instead of king square

Each feature set has a "Factorized" variant (suffix `^`) that adds virtual features separating king position from piece positions for improved training.

### Data Flow

1. **File Reading Thread(s)**: Read raw training entries from `.bin`/`.binpack` files
2. **Feature Extraction Workers**: Convert chess positions to sparse feature vectors
3. **Batch Assembly**: Group entries into `SparseBatch` with aligned arrays for efficient GPU transfer
4. **Python Consumption**: ctypes bindings in `nnue_dataset.py` (not in this repo) consume batches

Thread allocation: `concurrency` parameter is split 1:2 between reading threads and feature threads.

### Key Data Structures

**TrainingDataEntry**: Single training position
- `Position pos`: Chess board state
- `Move move`: Best move
- `int16_t score`: Engine evaluation
- `int16_t ply`: Move number
- `float result`: Game outcome (-1/0/+1)

**SparseBatch**: Batch of training data for neural network
- `int* white/black`: Sparse feature indices for each side
- `float* white_values/black_values`: Sparse feature values (mostly 1.0)
- `float* score/outcome/is_white`: Training labels
- `int* layer_stack_indices`: Bucket index based on piece count

**Position Orientation**:
- White's perspective: squares as-is
- Black's perspective: `flippedVertically().flippedHorizontally()` (180° rotation)
  - Note: Comment at line 51-52 indicates this is for Stockfish compatibility

## Custom Feature Set Development

To add a new feature set (example: HalfATA is partially implemented):

1. Define feature set struct with:
   - `static constexpr int INPUTS`: Total feature dimension
   - `static constexpr int MAX_ACTIVE_FEATURES`: Max active features per position
   - `static int feature_index(...)`: Map position components to feature index
   - `static std::pair<int, int> fill_features_sparse(...)`: Extract active features

2. Add factorized variant if needed (see `HalfATAFactorized` at lines 431-459)

3. Register in C API functions:
   - `get_sparse_batch_from_fens` (line 1121)
   - `create_sparse_batch_stream` (line 1204)

4. Implement `classify_attack_buckets()` or similar position classifier (line 399 stub)

5. Update Python-side feature set configuration

## Platform-Specific Notes

**BMI2 Support**: CMake auto-detects BMI2 CPU instruction support
- Enables faster bitboard operations via `_pdep_u64` intrinsic
- Detection logic at CMakeLists.txt:24-60
- Only enabled for non-AMD CPUs or AMD Zen 2+ (family ≥ 23)

**Export Macros**:
- x86_64: No special export needed
- Windows MSVC: `__declspec(dllexport)` and `__cdecl`
- Other: Default export

## Code Comments

This codebase contains extensive Chinese comments throughout training_data_loader.cpp explaining:
- Data loader architecture and design decisions
- Feature extraction logic for factorized features
- Multi-threading patterns (producer-consumer)
- SparseBatch memory layout

The Chinese comments are intentional documentation and should be preserved when modifying code.

## Dependencies

- C++17 required (set in CMakeLists.txt:19-20)
- CMake 3.10+
- Threads library (pthread on Unix)
- No external libraries required (chess logic is self-contained)

## Output

**Shared Library**: `libtraining_data_loader.so` (Linux/macOS) or `training_data_loader.dll` (Windows)
- Installed to CMAKE_INSTALL_PREFIX via `make install`
- Loaded by Python training scripts via ctypes

**Benchmark Executable**: `training_data_loader_benchmark`
- Only built with `CMAKE_BUILD_TYPE=PGO_Generate`
- Measures throughput (MPos/s, iterations/s, MB/s, bytes/position)
