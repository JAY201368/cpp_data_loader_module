# Minimal Test Harness - Setup Complete

## Overview

A minimal test harness has been successfully created for the training data loader. This harness allows you to test the data loader functionality with a small, controlled dataset of 20 chess positions.

## What Was Created

### 1. Test Data Generator ([generate_test_data.cpp](generate_test_data.cpp))
- C++ program that creates a `.bin` file with 20 chess positions
- Positions cover various game phases: openings, middlegames, and endgames
- Includes positions from: Italian Game, Sicilian Defense, French Defense, Queen's Gambit, etc.
- Each position includes score, ply count, and game result

### 2. Automated Test Script ([run_test_harness.sh](run_test_harness.sh))
- One-command test execution
- Builds generator, creates test data, builds benchmark, runs validation
- Provides clear success/failure feedback

### 3. Test Data Directory ([test_data/](test_data/))
- Contains generated test file: `chess_sample.bin` (800 bytes, 20 positions)
- Includes documentation: `README.md`

### 4. Build System Integration
- Updated [CMakeLists.txt](CMakeLists.txt) to build `generate_test_data` executable
- Works with existing build types (RelWithDebInfo, PGO_Generate, etc.)

## How to Use

### Quick Test (Recommended)
```bash
sh run_test_harness.sh
```

This runs the complete test workflow automatically.

### Manual Steps

1. **Build test generator:**
   ```bash
   mkdir -p build && cd build
   cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
   make generate_test_data
   cd ..
   ```

2. **Generate test data:**
   ```bash
   mkdir -p test_data
   ./build/generate_test_data test_data/chess_sample.bin
   ```

3. **Build benchmark:**
   ```bash
   cmake -S . -B build-pgo-generate -DCMAKE_BUILD_TYPE=PGO_Generate
   cmake --build ./build-pgo-generate --config PGO_Generate
   ```

4. **Run benchmark:**
   ```bash
   ./build-pgo-generate/training_data_loader_benchmark test_data/chess_sample.bin
   ```

## Validation Results

✅ **Test data generation**: Successfully creates 800-byte file with 20 positions
✅ **Data loader compilation**: Builds without errors
✅ **Benchmark execution**: Processes data correctly (2.7+ GPos/s throughput)
✅ **Cyclic reading**: Properly loops through small dataset

## Test Data Contents

The test dataset includes 20 positions:
- **11 opening positions**: e4, Sicilian, French, Caro-Kann, Queen's Gambit, King's Indian, Scandinavian, English, Reti, Bird's, Italian Game
- **6 middlegame positions**: Various tactical and strategic positions
- **3 endgame positions**: KR vs K, KQ vs K, KP vs K

Each entry includes:
- Chess position (FEN-encoded)
- Null move (placeholder for testing)
- Evaluation score (-100 to +100)
- Ply count (0-79)
- Game result (-1, 0, 1)

## Next Steps

### For Development
- Use `test_data/chess_sample.bin` for unit tests
- Modify `generate_test_data.cpp` to create custom test scenarios
- Add more positions for specific edge cases

### For Migration (Jungle)
- This baseline ensures current chess code works
- Use as reference when implementing Jungle game format
- Follow same pattern: generator → test data → benchmark

### For Feature Development
- Test new feature sets (e.g., HalfATA)
- Validate sparse batch generation
- Profile feature extraction performance

## File Locations

```
data_loader/
├── generate_test_data.cpp          # Test data generator
├── run_test_harness.sh             # Automated test script
├── CMakeLists.txt                  # Updated with generator build
├── test_data/
│   ├── chess_sample.bin            # Generated test data (800 bytes)
│   └── README.md                   # Detailed test data documentation
├── build/
│   └── generate_test_data          # Generator executable
└── build-pgo-generate/
    └── training_data_loader_benchmark  # Benchmark executable
```

## Customization

To add more test positions, edit `generate_test_data.cpp`:

```cpp
std::vector<std::string> test_fens = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    // Add your FEN strings here
    "your/custom/fen/string w KQkq - 0 1",
};
```

Then rebuild and regenerate:
```bash
cd build && make generate_test_data && cd ..
./build/generate_test_data test_data/chess_sample.bin
```

## Troubleshooting

**Issue**: "No rule to make target 'generate_test_data'"
- **Solution**: Delete and recreate build directory: `rm -rf build && mkdir build && cd build && cmake .. && cd ..`

**Issue**: Benchmark not found
- **Solution**: Build with PGO_Generate: `cmake -S . -B build-pgo-generate -DCMAKE_BUILD_TYPE=PGO_Generate`

**Issue**: Permission denied on script
- **Solution**: Make executable: `chmod +x run_test_harness.sh`

## References

- [JUNGLE_MIGRATION_PLAN.md](JUNGLE_MIGRATION_PLAN.md) - Phase 0: Pre-Migration Setup
- [test_data/README.md](test_data/README.md) - Detailed test data documentation
- [CLAUDE.md](CLAUDE.md) - Project overview and build instructions
