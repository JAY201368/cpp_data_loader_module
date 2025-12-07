# Data Format Hierarchy in NNUE Training Data Loader

Here's the relationship and hierarchy of these data formats:

## Visual Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                     CONCEPTUAL LEVEL                            │
├─────────────────────────────────────────────────────────────────┤
│  FEN (Human-Readable String)                                    │
│  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"     │
└────────────────────────┬────────────────────────────────────────┘
                         │ Position::fromFen()
                         ↓
┌────────────────────────────────────────────────────────────────┐
│                     RUNTIME/IN-MEMORY LEVEL                    │
├────────────────────────────────────────────────────────────────┤
│  TrainingDataEntry (C++ struct, ~100+ bytes)                   │
│  ┌───────────────────────────────────────────────────────┐     │
│  │ - chess::Position pos     (full bitboards, ~64 bytes) │     │
│  │ - chess::Move move        (4 bytes)                   │     │
│  │ - int16_t score           (2 bytes)                   │     │
│  │ - uint16_t ply            (2 bytes)                   │     │
│  │ - int16_t result          (2 bytes)                   │     │
│  └───────────────────────────────────────────────────────┘     │
└────────────────────────┬───────────────────────────────────────┘
                         │ trainingDataEntryToPackedSfenValue()
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                     SERIALIZATION LEVEL                         │
├─────────────────────────────────────────────────────────────────┤
│  SFEN (Shogi FEN) → PackedSfen → PackedSfenValue                │
│                                                                 │
│  PackedSfen (32 bytes) - Compressed position only               │
│  ┌─────────────────────────────────────────────┐                │
│  │ Bitpacked representation of board state     │                │
│  │ - Piece positions (compressed)              │                │
│  │ - Side to move, castling, en passant        │                │
│  └─────────────────────────────────────────────┘                │
│                         │                                       │
│                         ↓ (embedded in)                         │
│                                                                 │
│  PackedSfenValue (40 bytes) - Full training entry               │
│  ┌─────────────────────────────────────────────┐                │
│  │ - PackedSfen sfen        (32 bytes)         │                │
│  │ - int16_t score          (2 bytes)          │                │
│  │ - StockfishMove move     (2 bytes)          │                │
│  │ - uint16_t gamePly       (2 bytes)          │                │
│  │ - int8_t game_result     (1 byte)           │                │
│  │ - uint8_t padding        (1 byte)           │                │
│  └─────────────────────────────────────────────┘                │
└────────────────────────┬────────────────────────────────────────┘
                         │ Write to disk
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                     FILE STORAGE LEVEL                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  .bin File (Uncompressed)                                       │
│  ┌─────────────────────────────────────────────┐                │
│  │ [PackedSfenValue][PackedSfenValue][...]     │                │
│  │  40 bytes each, sequential, no compression  │                │
│  └─────────────────────────────────────────────┘                │
│  Simple format: Direct binary dump                              │
│  20 positions = 800 bytes (40 × 20)                             │
│                                                                 │
│  .binpack File (Compressed)                                     │
│  ┌─────────────────────────────────────────────┐                │
│  │ [Chunk Header][PackedEntry][Movelist]...    │                │
│  │ Smart compression with delta encoding       │                │
│  └─────────────────────────────────────────────┘                │
│  Advanced format:                                               │
│  - Groups consecutive moves from same game                      │
│  - Delta-encodes scores and moves                               │
│  - Typical compression: 3-5x smaller than .bin                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Detailed Explanation

### 1. FEN (Forsyth-Edwards Notation)

- Type: Human-readable string format
- Purpose: Standard chess notation for sharing positions
- Size: Variable (~50-100 characters)
- Example: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
- Contains:
  - Piece positions (rank by rank)
  - Side to move (w/b)
  - Castling rights (KQkq)
  - En passant square
  - Halfmove clock
  - Fullmove number
- Usage: Input/output, debugging, visualization
- Location in code: Used by Position::fromFen() and Position::fen()

### 2. SFEN (Shogi FEN)

- Type: Conceptual - refers to the Shogi (Japanese chess) FEN format
- Purpose: Similar to FEN but for Shogi; term used in this codebase for compressed formats
- Note: Despite the name, this codebase uses it for standard chess, borrowed from Stockfish NNUE training infrastructure
  
### 3. TrainingDataEntry (In-Memory Representation)

```cpp
struct TrainingDataEntry {
    chess::Position pos;  // Full position with bitboards (~64 bytes)
    chess::Move move;     // Best move (4 bytes)
    int16_t score;        // Evaluation (-32768 to +32767 centipawns)
    uint16_t ply;         // Move number from game start
    int16_t result;       // Game outcome: -1 (loss), 0 (draw), 1 (win)
}
```

- Size: ~100+ bytes (depends on Position implementation)
- Type: C++ struct (in-memory)
- Purpose: Working format used by C++ code during data loading
- Contains: Full chess position state plus training labels
- Usage: Runtime processing, feature extraction, batch formation
- Location: lib/nnue_training_data_formats.h:6894
- Key Methods:
  - isValid(): Check if move is legal
  - isCapturingMove(): Check if move captures
  - win_rate_model(): Calculate win/draw/loss probabilities
  
### 4. PackedSfen (Compressed Position Only)

```cpp
struct PackedSfen {
    uint8_t data[32];  // Bit-packed board representation
}
```

- Type: Binary serialization format
- Size: 32 bytes (fixed)
- Purpose: Space-efficient position encoding
- Contains: ONLY the board position (no move, score, or metadata)
- Compression: Uses bitstreams to pack pieces into minimal space
  - Each piece encoded in ~4 bits
  - Empty squares efficiently compressed
  - Special encoding for castling rights, en passant
- Location: lib/nnue_training_data_formats.h:6407
- Packing: Done by SfenPacker class (bitstream operations)
  
### 5. PackedSfenValue (Complete Training Sample)

```cpp
struct PackedSfenValue {  // 40 bytes total
    PackedSfen sfen;              // 32 bytes - position
    int16_t score;                // 2 bytes  - evaluation
    StockfishMove move;           // 2 bytes  - compressed move
    uint16_t gamePly;             // 2 bytes  - ply count
    int8_t game_result;           // 1 byte   - outcome
    uint8_t padding;              // 1 byte   - alignment
}
```

- Type: Binary serialization format
- Size: 40 bytes (fixed)
- Purpose: Complete training data point ready for disk storage
- Contains: Everything needed for NNUE training
- Location: lib/nnue_training_data_formats.h:6412
- Conversion:
  - trainingDataEntryToPackedSfenValue(): In-memory → disk format
  - packedSfenValueToTrainingDataEntry(): Disk format → in-memory
  
### 6. .bin File (Uncompressed Binary File)

- Type: File format
- Structure: Sequential PackedSfenValue records
- Size: 40 bytes per position (fixed)
  - Example: 1 million positions = 40 MB
  - Your test file: 20 positions = 800 bytes
- Compression: None (direct binary dump)
- Pros:
  - Simple to write/read
  - Fast random access
  - Easy to implement
- Cons:
  - Large file sizes
  - No compression
- Reading: BinSfenInputStream
- Writing: Direct binary write of PackedSfenValue (as in your generate_test_data.cpp)
  
### 7. .binpack File (Compressed Binary Format)

- Type: Advanced file format with compression
- Structure: Chunked format with delta encoding

```
[Chunk 1: ~1MB]
  [PackedEntry 1] [Movelist: next N moves & scores]
  [PackedEntry 2] [Movelist: next M moves & scores]
  ...
[Chunk 2: ~1MB]
  ...
```

- Size: Typically 3-5x smaller than .bin
  - Example: 1 million positions = 8-13 MB (vs 40 MB for .bin)
- Compression Techniques:
  - Delta encoding: Store differences between consecutive positions from same game
  - Movelist packing: Group consecutive moves together
  - Move continuation detection: Positions from same game stored compactly
- Pros:
  - Much smaller files
  - Better for distributing large datasets
  - Exploits game structure (consecutive positions are similar)
- Cons:
  - More complex to implement
  - No random access (must read sequentially)
  - Slightly slower decompression
- Reading: BinpackSfenInputStream / BinpackSfenInputParallelStream
- Writing: CompressedTrainingDataEntryWriter

### Data Flow During Training

1. GENERATION (by chess engine):
   Game playing → FEN strings → .bin/.binpack files

2. LOADING (this data loader):
   .bin/.binpack → PackedSfenValue → TrainingDataEntry

3. FEATURE EXTRACTION:
   TrainingDataEntry → Sparse features (HalfKP, HalfKA, etc.)

4. TRAINING (Python):
   Sparse features → Neural network → Gradients → Updated weights

### Size Comparison

| Format | Size per Position | 1M Positions	| Notes |
| FEN string |	~70 bytes |	~70 MB |	Human-readable |
| TrainingDataEntry	| ~100+ bytes |	~100+ MB |	In-memory, includes Position object |
| PackedSfenValue |	40 bytes |	40 MB	| Compressed but complete |
| .bin file |	40 bytes | 40 MB | Direct PackedSfenValue dump |
| .binpack file | 8-13 bytes | 8-13 MB | Delta-encoded, 3-5x compression |

### When to Use Each Format

- FEN: Human interaction, debugging, position input
- TrainingDataEntry: Runtime processing in C++
- PackedSfenValue: Intermediate serialization format
- .bin: Simple storage, testing, small datasets
- .binpack: Production training data, large datasets, distribution

### Key Takeaway

The hierarchy is:
- FEN (human) →
- TrainingDataEntry (runtime) →
- PackedSfenValue (serialization) →
- .bin (simple file) or .binpack (compressed file)
  
Your test harness generates .bin files (simplest format) with 20 positions = 800 bytes, perfect for testing!

# Updated Hierarchy: Move Representations

```
┌─────────────────────────────────────────────────────────────────┐
│                     RUNTIME LEVEL (C++)                         │
├─────────────────────────────────────────────────────────────────┤
│  chess::Move (4 bytes)                                          │
│  ┌───────────────────────────────────────────────────────┐      │
│  │ Square from;          // 1 byte (0-63)                │      │
│  │ Square to;            // 1 byte (0-63)                │      │
│  │ MoveType type;        // 1 byte (Normal/Castle/etc)   │      │
│  │ Piece promotedPiece;  // 1 byte (for promotions)      │      │
│  └───────────────────────────────────────────────────────┘      │
│  Used in: TrainingDataEntry, runtime move generation            │
└────────────────────────┬────────────────────────────────────────┘
                         │
        ┌────────────────┴─────────────────┐
        │                                  │
        ↓                                  ↓
┌───────────────────────┐      ┌───────────────────────────────┐
│ SERIALIZATION PATH 1  │      │ SERIALIZATION PATH 2          │
│ (This codebase)       │      │ (Stockfish compatibility)     │
├───────────────────────┤      ├───────────────────────────────┤
│                       │      │                               │
│ chess::CompressedMove │      │ nodchip::StockfishMove        │
│ (2 bytes, 16 bits)    │      │ (2 bytes, 16 bits)            │
│                       │      │                               │
│ Bit layout:           │      │ Bit layout:                   │
│ ┌─────────────────┐   │      │ ┌─────────────────┐           │
│ │ [15:14] type    │   │      │ │ [15:14] moveFlag│           │
│ │ [13:8]  from sq │   │      │ │ [13:12] promoIdx│           │
│ │ [7:2]   to sq   │   │      │ │ [11:6]  from sq │           │
│ │ [1:0]   promo   │   │      │ │ [5:0]   to sq   │           │
│ └─────────────────┘   │      │ └─────────────────┘           │
│                       │      │                               │
│ Used in:              │      │ Used in:                      │
│ - PackedSfen          │      │ - PackedSfenValue             │
│ - binpack format      │      │ - .bin file format            │
│ - Position encoding   │      │ - Stockfish compatibility     │
└───────────────────────┘      └───────────────────────────────┘
```

## Detailed Explanation

### chess::Move (Runtime, 4 bytes)

- Location: lib/nnue_training_data_formats.h:1551
- Size: 4 bytes
- Purpose: Full move representation for runtime use
- Fields:
  - from: Source square (0-63)
  - to: Destination square (0-63)
  - type: Move type (Normal, Castle, EnPassant, Promotion)
  - promotedPiece: Which piece to promote to (if promotion)

### chess::CompressedMove (Serialization, 2 bytes)

- Location: lib/nnue_training_data_formats.h:1611
- Size: 2 bytes (16 bits)
- Purpose: Space-efficient move encoding for this codebase's internal formats
- Bit Layout (Big Endian):
  - Bits 15-14: Move type (2 bits)
    - 00 = Normal
    - 01 = Promotion  
    - 10 = EnPassant
    - 11 = Castle
  - Bits 13-8:  From square (6 bits, 0-63)
  - Bits 7-2:   To square (6 bits, 0-63)
  - Bits 1-0:   Promotion type (2 bits, if promotion)
    - 00 = Knight
    - 01 = Bishop
    - 02 = Rook
    - 03 = Queen
- Used In:
  - PackedSfen structure (position compression)
  - .binpack file format
  - Position serialization/deserialization
- Conversion:
  - Move::compress() → CompressedMove line 1747
  - Constructor: CompressedMove(Move) line 1638

### nodchip::StockfishMove (Stockfish compatibility, 2 bytes)

- Location: lib/nnue_training_data_formats.h:6330
- Size: 2 bytes (16 bits)
- Purpose: Stockfish-compatible move encoding for .bin files
- Bit Layout (Different order!):
  - Bits 15-14: Move flag (2 bits)
    - 00 = Normal
    - 01 = Promotion
    - 10 = EnPassant
    - 11 = Castle
  - Bits 13-12: Promotion index (2 bits, if promotion)
    - 00 = Knight
    - 01 = Bishop
    - 10 = Rook
    - 11 = Queen
  - Bits 11-6:  From square (6 bits, 0-63)
  - Bits 5-0:   To square (6 bits, 0-63)
- Used In:
  - PackedSfenValue.move field line 6422
  - .bin file format
  - Compatibility with Stockfish NNUE training data
- Conversion:
  - StockfishMove::fromMove(Move) line 6332
  - toMove() → Move line 6360

## Why Two Different Formats?

- CompressedMove: Used internally by this codebase for position compression
- StockfishMove: Used for compatibility with Stockfish's training data format

The bit layouts are different because they serve different purposes and maintain compatibility with different systems.

## Where They Appear in the Main Hierarchy

Here's the updated complete hierarchy with moves:

```
TrainingDataEntry (runtime)
├── chess::Position pos
├── chess::Move move ◄───────────────┐ [4 bytes runtime]
├── int16_t score                    │
├── uint16_t ply                     │
└── int16_t result                   │
                                     │
         ↓ serialize                 │
                                     │
PackedSfenValue (40 bytes)           │
├── PackedSfen sfen                  │
│   └── Uses chess::CompressedMove ◄─┤ [2 bytes, internal format]
├── int16_t score                    │
├── StockfishMove move ◄─────────────┘ [2 bytes, Stockfish format]
├── uint16_t gamePly
├── int8_t game_result
└── uint8_t padding
```

## Summary

CompressedMove is a serialization format that sits between the runtime Move (4 bytes) and disk storage:

- Runtime: chess::Move (4 bytes) - used during execution
- Serialization (internal): chess::CompressedMove (2 bytes) - used in PackedSfen/binpack
- Serialization (Stockfish): nodchip::StockfishMove (2 bytes) - used in PackedSfenValue/.bin files

Both compressed formats reduce 4 bytes down to 2 bytes (50% compression) by packing the move information into bit fields, but they use different bit layouts for compatibility with different systems.

## Can You Remove CompressedMove?

Short Answer: Yes, you can remove CompressedMove if you're willing to make storage compromises, BUT it depends on which formats you want to support.

### Analysis of CompressedMove Usage

Where CompressedMove is Used:
- .binpack format (compressed file format) - line 7437
- Used in PackedTrainingDataEntry (32 bytes)
- Critical for the compression algorithm
- Position history/undo - lines 1898, 1945
- CompressedReverseMove stores move history for undoing moves
- Used when making/unmaking moves during position traversal
- Binpack reader/writer - Used throughout delta encoding
  
Where CompressedMove is NOT Used:
- .bin format - Uses StockfishMove instead (in PackedSfenValue)
- TrainingDataEntry - Uses full chess::Move (4 bytes)
- Feature extraction - Works with chess::Move

### Your Options

Option 1: Keep Only .bin Format (Simple Storage)

Remove CompressedMove: ✅ YES, you can remove it

```
// What you keep:
- chess::Move (4 bytes, runtime)
- nodchip::StockfishMove (2 bytes, for .bin files)
- PackedSfenValue (40 bytes)

// What you lose:
- .binpack format support
- 3-5x compression
- Position history compression
```

Storage Impact:
- 1 million positions: 40 MB (.bin) vs 8-13 MB (.binpack)
- Your 20-position test file: 800 bytes (no change)

Code Changes Required:

```
// Remove these:
- struct CompressedMove (line 1611)
- struct CompressedReverseMove (line 1882)  
- PackedTrainingDataEntry / binpack format code
- CompressedTrainingDataEntryReader/Writer

// Keep these:
- struct Move (line 1551)
- struct StockfishMove (line 6330)
- PackedSfenValue (line 6412)
- BinSfenInputStream (for .bin files)
```

Pros:
- ✅ Simpler codebase (remove ~1500 lines of compression code)
- ✅ Easier to understand and maintain
- ✅ Still have 10x compression vs FEN strings
- ✅ Your test harness already uses this format

Cons:
- ❌ Larger file sizes (3-5x bigger than .binpack)
- ❌ Can't read existing .binpack training datasets
- ❌ Less efficient for distributing large datasets
