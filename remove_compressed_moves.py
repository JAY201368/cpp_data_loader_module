#!/usr/bin/env python3
"""
Script to remove CompressedMove, CompressedPosition, and all binpack infrastructure
from nnue_training_data_formats.h to simplify to .bin-only format.
"""

import sys

def remove_line_ranges(content_lines, ranges_to_remove):
    """
    Remove specified line ranges from content.
    ranges_to_remove: list of (start_line, end_line) tuples (1-indexed, inclusive)
    """
    # Convert to 0-indexed and sort in reverse order to remove from back to front
    ranges_0indexed = [(start-1, end) for start, end in ranges_to_remove]
    ranges_0indexed.sort(reverse=True)

    result_lines = content_lines[:]
    for start_idx, end_idx in ranges_0indexed:
        del result_lines[start_idx:end_idx]

    return result_lines

def process_formats_h():
    """Process lib/nnue_training_data_formats.h"""
    filepath = "lib/nnue_training_data_formats.h"

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Original file: {len(lines)} lines")

    # Define all ranges to remove (line numbers are 1-indexed, inclusive)
    # Ordered from largest line numbers to smallest to avoid offset issues
    ranges_to_remove = [
        # Phase 3: Remove binpack namespace content (keep nodchip, TrainingDataEntry, helpers)
        (8153, 8224),  # convertPlainToBin
        (8097, 8151),  # convertBinToPlain
        (8050, 8095),  # convertBinpackToBin
        (8008, 8048),  # convertBinToBinpack
        (7960, 8005),  # convertBinpackToPlain
        (7901, 7958),  # convertPlainToBinpack
        (7894, 7899),  # emitBinEntry
        (7871, 7892),  # emitPlainEntry
        (7623, 7869),  # CompressedTrainingDataEntryParallelReader
        (7535, 7621),  # CompressedTrainingDataEntryReader
        (7452, 7533),  # CompressedTrainingDataEntryWriter
        (7429, 7450),  # unpackEntry
        (7404, 7427),  # packEntry
        (7235, 7401),  # PackedMoveScoreList
        (7018, 7233),  # PackedMoveScoreListReader
        (7010, 7014),  # usedBitsSafe
        (7005, 7008),  # PackedTrainingDataEntry
        (6773, 6867),  # CompressedTrainingDataFile
        # Phase 3: Remove CompressedPosition
        (5472, 6309),  # CompressedPosition::decompress() - HUGE! 838 lines
        (5438, 5470),  # Position::compress() implementation
        (4704, 4704),  # CompressedPosition::decompress() declaration
        (4632, 4702),  # struct CompressedPosition
        (4614, 4614),  # Position::compress() declaration
        (4405, 4405),  # CompressedPosition forward declaration
        # Phase 2: Remove CompressedReverseMove
        (1951, 1954),  # ReverseMove::compress() implementation
        (1949, 1949),  # static_assert(sizeof(CompressedReverseMove) == 4)
        (1882, 1947),  # struct CompressedReverseMove
        (1864, 1864),  # ReverseMove::compress() declaration
        (1833, 1833),  # CompressedReverseMove forward declaration
        # Phase 2: Remove CompressedMove
        (1747, 1750),  # Move::compress() implementation
        (1745, 1745),  # static_assert(sizeof(CompressedMove) == 2)
        (1611, 1743),  # struct CompressedMove - 133 lines
        (1571, 1571),  # Move::compress() declaration
        (1547, 1547),  # CompressedMove forward declaration
    ]

    # Remove the lines
    result_lines = remove_line_ranges(lines, ranges_to_remove)

    print(f"After removal: {len(result_lines)} lines")
    print(f"Removed: {len(lines) - len(result_lines)} lines")

    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(result_lines)

    print(f"✓ Updated {filepath}")

def process_loader_cpp():
    """Process training_data_loader.cpp"""
    filepath = "training_data_loader.cpp"

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Update comments and remove binpack namespace
    content = content.replace(
        "处理加载训练数据(.bin, .binpack)",
        "处理加载训练数据(.bin)"
    )
    content = content.replace(
        "using namespace binpack;",
        "// binpack namespace removed - using .bin format only"
    )
    content = content.replace(
        "g++ -std=c++20 -g3 -O3 -DNDEBUG -DBENCH -march=native training_data_loader.cpp && ./a.out /path/to/binpack",
        "g++ -std=c++20 -g3 -O3 -DNDEBUG -DBENCH -march=native training_data_loader.cpp && ./a.out /path/to/file.bin"
    )

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✓ Updated {filepath}")

def process_test_generator():
    """Process generate_test_data.cpp"""
    filepath = "generate_test_data.cpp"

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove binpack namespace usage
    content = content.replace(
        "using namespace binpack;",
        "// binpack namespace removed - using .bin format only"
    )

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✓ Updated {filepath}")

def main():
    print("="*60)
    print("Removing CompressedMove and binpack infrastructure")
    print("="*60)
    print()

    try:
        print("Phase 2 & 3: Processing lib/nnue_training_data_formats.h...")
        process_formats_h()
        print()

        print("Phase 4: Processing training_data_loader.cpp...")
        process_loader_cpp()
        print()

        print("Phase 4: Processing generate_test_data.cpp...")
        process_test_generator()
        print()

        print("="*60)
        print("✓ All files processed successfully!")
        print("="*60)
        print()
        print("Next steps:")
        print("1. Compile: sh compile_data_loader.bat")
        print("2. Test: sh run_test_harness.sh")

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
