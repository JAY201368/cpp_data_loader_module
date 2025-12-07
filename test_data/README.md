# Test Harness Documentation

This directory contains a minimal test harness for the training data loader, consisting of 20 chess positions covering various openings and game phases.

## Files

- **generate_test_data.cpp**: C++ program that generates a small test `.bin` file with sample chess positions
- **run_test_harness.sh**: Shell script that automates the entire test process
- **test_data/**: Directory containing generated test data files

## Quick Start

Run the complete test harness:

```bash
sh run_test_harness.sh
```

This will:
1. Build the test data generator
2. Generate `test_data/chess_sample.bin` with 20 chess positions
3. Build the benchmark executable
4. Run the benchmark on the test data

## Manual Steps

### 1. Build the Test Data Generator

```bash
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
make generate_test_data
cd ..
```

### 2. Generate Test Data

```bash
mkdir -p test_data
./build/generate_test_data test_data/chess_sample.bin
```

This creates a `.bin` file with 20 positions (~800 bytes).

### 3. Build the Benchmark

```bash
cmake -S . -B build-pgo-generate -DCMAKE_BUILD_TYPE=PGO_Generate
cmake --build ./build-pgo-generate --config PGO_Generate
```

### 4. Run Benchmark

```bash
./build-pgo-generate/training_data_loader_benchmark test_data/chess_sample.bin
```

## Test Data Contents

The test dataset includes 20 chess positions covering:

- **Opening positions**:
  - Starting position
  - Italian Game
  - Sicilian Defense
  - French Defense
  - Caro-Kann Defense
  - Queen's Gambit
  - King's Indian setup
  - Scandinavian Defense
  - English Opening
  - Reti Opening
  - Bird's Opening

- **Middlegame positions**:
  - Complex tactical positions
  - Typical middlegame structures

- **Endgame positions**:
  - KR vs K (King + Rook vs King)
  - KQ vs K (King + Queen vs King)
  - KP vs K (King + Pawn vs King)

Each position includes:
- Position state (FEN)
- Placeholder move (not meaningful for testing)
- Evaluation score (-100 to +100)
- Ply count (0-79)
- Game result (-1, 0, or 1)

## Customizing Test Data

To modify the test positions, edit [generate_test_data.cpp](generate_test_data.cpp):

1. Add/remove FEN strings in the `test_fens` vector
2. Modify the move generation logic (currently uses dummy moves)
3. Adjust score/ply/result calculation
4. Rebuild and regenerate:

```bash
cd build && make generate_test_data && cd ..
./build/generate_test_data test_data/chess_sample.bin
```

## File Format

The `.bin` format uses `nodchip::PackedSfenValue` structure:
- Compressed chess position (32 bytes)
- Move (2 bytes)
- Score (2 bytes)
- Ply (2 bytes)
- Result (2 bytes)
- Padding (40 bytes total per position)

See [lib/nnue_training_data_formats.h](lib/nnue_training_data_formats.h) for format details.

## Baseline Validation

Expected output from benchmark:
- Successfully loads all 20 positions
- Cyclic reading enabled (will loop through data)
- Throughput metrics displayed (MPos/s, It/s, MB/s)

Sample output:
```
Iter:       30       Time(s):     2.341       MPos/s:     0.210       It/s:     12.8       MB/s:     42.3       B/pos:  204.8
```

## Next Steps After Testing

Once the test harness validates your setup:

1. **Unit Testing**: Use `test_data/chess_sample.bin` in unit tests
2. **Feature Development**: Test new feature sets (HalfATA, etc.)
3. **Performance Profiling**: Use PGO build process
4. **Full Dataset**: Run on production training data

## Troubleshooting

**Build errors**: Ensure C++17 compiler and CMake 3.10+
**Benchmark not found**: Run with `CMAKE_BUILD_TYPE=PGO_Generate`
**File read errors**: Check `test_data/` directory exists and has write permissions
