#!/bin/bash
# Minimal test harness for training data loader
# This script:
# 1. Builds the test data generator
# 2. Creates a small test dataset
# 3. Compiles the data loader
# 4. Runs the benchmark on the test data

set -e  # Exit on error

echo "=== Training Data Loader Test Harness ==="
echo

# Create test_data directory if it doesn't exist
mkdir -p test_data

# Step 1: Build the test data generator
echo "Step 1: Building test data generator..."
if [ ! -d "build" ]; then
    echo "Build directory not found. Creating..."
    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
    cd ..
fi

cd build
make generate_test_data
cd ..

# Step 2: Generate test data
echo
echo "Step 2: Generating test data..."
./build/generate_test_data test_data/chess_sample.bin

# Verify the file was created
if [ ! -f "test_data/chess_sample.bin" ]; then
    echo "ERROR: Failed to generate test data file"
    exit 1
fi

echo "Test data file created: test_data/chess_sample.bin"
ls -lh test_data/chess_sample.bin

# Step 3: Build benchmark (if not already built)
echo
echo "Step 3: Ensuring benchmark is built..."
if [ ! -f "build-pgo-generate/training_data_loader_benchmark" ]; then
    echo "Building benchmark executable..."
    cmake -S . -B build-pgo-generate -DCMAKE_BUILD_TYPE=PGO_Generate
    cmake --build ./build-pgo-generate --config PGO_Generate
fi

# Step 4: Run benchmark on test data
echo
echo "Step 4: Running benchmark on test data..."
echo "This will verify the data loader can read and process the data."
echo
./build-pgo-generate/training_data_loader_benchmark test_data/chess_sample.bin

echo
echo "=== Test Complete ==="
echo "✓ Test data generated successfully"
echo "✓ Data loader compiled successfully"
echo "✓ Benchmark ran successfully"
echo
echo "Next steps:"
echo "  - Use test_data/chess_sample.bin for unit tests"
echo "  - Modify generate_test_data.cpp to create custom test scenarios"
echo "  - Run: ./build-pgo-generate/training_data_loader_benchmark test_data/chess_sample.bin"
