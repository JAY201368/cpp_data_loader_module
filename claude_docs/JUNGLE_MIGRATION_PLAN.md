# Jungle Chess Migration Plan

## Executive Summary

This document outlines a **continuous integration approach** to migrate the chess NNUE training data loader to Jungle chess (斗兽棋/Animal Chess). The strategy focuses on incremental changes with validation at each step to prevent cascading errors.

## Key Differences: Chess vs Jungle Chess

### Board Structure
- **Chess**: 8×8 (64 squares)
- **Jungle**: 7×9 (63 squares)
  - Includes: 2 dens (兽穴), 6 traps (陷阱), 2 rivers (河流, 3×2 squares each)

### Pieces
- **Chess**: 6 piece types × 2 colors = 12 piece-color combinations (excluding king in some feature sets: 10)
- **Jungle**: 8 piece types × 2 colors = 16 piece-color combinations
  - Elephant (象), Lion (狮), Tiger (虎), Leopard (豹), Wolf (狼), Dog (狗), Cat (猫), Rat (鼠)

### Critical Differences
1. **No King**: Jungle chess has dens instead of kings
2. **Special Terrain**: Traps, rivers, and dens affect movement and capture rules
3. **Piece Power Hierarchy**: Rat can capture Elephant; animals in traps can be captured by any piece
4. **Move Types**: No castling, en passant, or promotion

## Migration Strategy: 6-Phase Approach

### Phase 0: Pre-Migration Setup ✓
**Goal**: Establish testing infrastructure and baseline

**Tasks**:
- [ ] Create git branch `jungle-migration`
- [ ] Document current test data format (example .bin/.binpack files)
- [ ] Create minimal test harness (10-20 position test file)
- [ ] Establish baseline: current code compiles and runs on chess data

**Validation**:
```bash
git checkout -b jungle-migration
make clean && sh compile_data_loader.bat
./build/training_data_loader_benchmark test_data/chess_sample.binpack
```

**Deliverable**: Working baseline on new branch

---

### Phase 1: Core Data Structure Adaptation
**Goal**: Modify fundamental types in `lib/nnue_training_data_formats.h` for Jungle chess geometry

**Estimated Scope**: ~200-400 lines changed in 8000+ line file

#### 1.1: Square and Board Geometry (Week 1)

**Changes**:
```cpp
// Current (chess.h equivalent section)
namespace chess {
    enum FileEnum : uint8_t { fileA, fileB, ..., fileH }; // 8 files
    enum RankEnum : uint8_t { rank1, rank2, ..., rank8 }; // 8 ranks

    struct Square {
        static constexpr int Count = 64;
        // ...
    }
}

// Target (jungle)
namespace jungle {
    enum FileEnum : uint8_t { fileA, fileB, ..., fileG }; // 7 files
    enum RankEnum : uint8_t { rank1, rank2, ..., rank9 }; // 9 ranks

    struct Square {
        static constexpr int Count = 63; // 7×9
        // Note: May need special encoding for terrain types
    }
}
```

**Modified Entities**:
- `Square::Count`: 64 → 63
- File constants: fileA-fileH → fileA-fileG
- Rank constants: rank1-rank8 → rank1-rank9
- Bitboard operations (if 64-bit assumption is hardcoded)

**Critical Search Patterns**:
```bash
# Find all hardcoded 64 assumptions
grep -n "\b64\b" lib/nnue_training_data_formats.h

# Find all 8×8 assumptions
grep -n "\b8\b" lib/nnue_training_data_formats.h | grep -i "rank\|file\|square"
```

**Validation**:
- [ ] Code compiles (may not link yet)
- [ ] Unit test: Square enumeration covers exactly 63 values
- [ ] Unit test: File/rank arithmetic works for 7×9 board

**Commit**: `Phase 1.1: Adapt Square/Board geometry for 7×9 Jungle board`

#### 1.2: Piece Type Adaptation (Week 1)

**Changes**:
```cpp
// Current
namespace chess {
    enum class PieceType : uint8_t {
        None, Pawn, Knight, Bishop, Rook, Queen, King
    };
}

// Target
namespace jungle {
    enum class PieceType : uint8_t {
        None, Rat, Cat, Dog, Wolf, Leopard, Tiger, Lion, Elephant
    };

    // Power hierarchy: Elephant(8) > Lion(7) > Tiger(6) > Leopard(5)
    //                  > Wolf(4) > Dog(3) > Cat(2) > Rat(1)
    // Special: Rat can capture Elephant
}
```

**Modified Entities**:
- `PieceType` enum values
- `Piece` struct and methods
- Piece count constants (6 → 8 types)
- Piece character representations (for FEN-like notation)

**Validation**:
- [ ] All 8 piece types compile
- [ ] Unit test: Piece enumeration and color combinations (16 total)
- [ ] Unit test: Piece string parsing/serialization

**Commit**: `Phase 1.2: Replace chess pieces with Jungle piece types`

#### 1.3: Move Type Simplification (Week 1-2)

**Changes**:
```cpp
// Current
enum class MoveType { Normal, Promotion, EnPassant, Castle };

// Target
enum class MoveType { Normal }; // Jungle has only normal moves
// Consider: May need Jump (for Lion/Tiger river crossing)
```

**Modified Entities**:
- Remove `MoveType::Promotion`, `EnPassant`, `Castle`
- Simplify `Move` struct (remove `promotedPiece` field)
- Update `StockfishMove` in nodchip namespace (lines 6330-6400)

**Validation**:
- [ ] Move encoding/decoding works for simple moves
- [ ] No compilation errors from removed move types
- [ ] Unit test: Move serialization round-trip

**Commit**: `Phase 1.3: Simplify MoveType for Jungle chess (no castling/promotion)`

#### 1.4: Position and Board State (Week 2)

**Changes**:
- Update `Board` and `Position` structs (around line 4429)
- Modify bitboard operations for 63 squares
- Replace king square tracking with den square tracking (if needed for features)
- Add terrain encoding (dens, traps, rivers)

**Critical**: Jungle board has special squares:
```
Rank 9: Red den at (D9)
Rank 8: Red traps at (C8, E8, D7)
Rank 4-6: Rivers at (B4-C6, E4-F6) - 6 water squares
Rank 3: Black traps at (D3, C2, E2)
Rank 1: Black den at (D1)
```

**Options for Terrain**:
1. **Implicit**: Hardcode terrain rules in move generation
2. **Explicit**: Add `SquareType` field to distinguish land/water/trap/den

**Validation**:
- [ ] Position initialization works
- [ ] Piece placement/removal works
- [ ] FEN-like string parsing (Jungle format)
- [ ] Unit test: Starting position loads correctly

**Commit**: `Phase 1.4: Adapt Position struct for Jungle board state`

---

### Phase 2: Move Generation and Game Logic
**Goal**: Implement Jungle chess move rules

**Estimated Scope**: ~500-800 lines (move generation section)

#### 2.1: Basic Move Generation (Week 2-3)

**Changes** (around line 5000-6000 in nnue_training_data_formats.h):
- Rewrite `forEachPseudoLegalMove` for Jungle rules
- Implement piece-specific movement:
  - **All pieces**: One square orthogonally (no diagonals except river jumps)
  - **Lion/Tiger**: Can jump over river (3 squares orthogonally if no piece in between)
  - **Rat**: Can enter river; can attack from river to land

**Validation**:
- [ ] Move generation produces legal moves for each piece type
- [ ] Unit test: Known positions with expected move counts
- [ ] Unit test: River jump mechanics for Lion/Tiger
- [ ] Unit test: Rat river movement

**Commit**: `Phase 2.1: Implement basic Jungle chess move generation`

#### 2.2: Capture Rules (Week 3)

**Changes**:
- Implement power hierarchy captures
- Special rules:
  - Rat (1) can capture Elephant (8)
  - Pieces in opponent's trap lose power (can be captured by any piece)
  - Pieces cannot enter their own den
  - Game ends when piece enters opponent's den

**Validation**:
- [ ] Unit test: Correct captures based on power hierarchy
- [ ] Unit test: Rat-Elephant special capture
- [ ] Unit test: Trap capture mechanics
- [ ] Unit test: Den entry detection

**Commit**: `Phase 2.2: Implement Jungle chess capture rules and win condition`

#### 2.3: Testing Integration (Week 3)

**Changes**:
- Update `forEachLegalMove` to filter pseudolegal moves
- Ensure Position::isCheck() equivalent (or remove if not applicable)
- Test with complete game scenarios

**Validation**:
- [ ] Integration test: Play through sample Jungle game
- [ ] Verify no illegal moves generated
- [ ] Performance test: Move generation speed

**Commit**: `Phase 2.3: Complete move generation and validate game logic`

---

### Phase 3: Binary Format Adaptation
**Goal**: Modify .bin/.binpack format for Jungle chess data

**Estimated Scope**: ~200-300 lines (binpack namespace, ~line 6311)

#### 3.1: PackedSfen Format (Week 4)

**Changes** (nodchip::PackedSfen around line 6407):
- Current: Stockfish packed format (256 bits)
- Target: Jungle packed format

**Key Differences**:
```cpp
// Current: 256 bits for chess position
struct PackedSfen {
    uint8_t data[24];  // Piece placement
    uint8_t unused[8]; // Padding
};

// Target: Adjusted for Jungle
struct PackedSfen {
    uint8_t data[20];  // 63 squares, ~5 bits per square max
    // May need different packing density
};
```

**Validation**:
- [ ] Serialize/deserialize Jungle positions correctly
- [ ] Round-trip test: Position → PackedSfen → Position
- [ ] Test with boundary cases (all pieces, minimal pieces)

**Commit**: `Phase 3.1: Adapt PackedSfen format for Jungle chess positions`

#### 3.2: TrainingDataEntry (Week 4)

**Changes** (around line 6894):
```cpp
struct TrainingDataEntry {
    jungle::Position pos;  // Updated to jungle::Position
    jungle::Move move;     // Simplified move
    int16_t score;         // Keep same
    uint16_t ply;          // Keep same
    int8_t result;         // Keep same
    // ... other fields
};
```

**Validation**:
- [ ] TrainingDataEntry serialization works
- [ ] Compatibility test: Read/write .binpack files
- [ ] Create sample Jungle training data file (10 positions)

**Commit**: `Phase 3.2: Update TrainingDataEntry for Jungle chess`

#### 3.3: File I/O Streams (Week 4)

**Changes**:
- Verify `BinpackSfenInputStream` works with new format
- Update compression/decompression if needed
- Test parallel reading with new data

**Validation**:
- [ ] Successfully read Jungle .binpack files
- [ ] Parallel reading maintains correctness
- [ ] Performance: Compare throughput with chess data

**Commit**: `Phase 3.3: Validate file I/O with Jungle training data`

---

### Phase 4: Feature Set Implementation
**Goal**: Implement Jungle-specific feature sets (HalfATA and others)

**Estimated Scope**: ~400-600 lines (training_data_loader.cpp)

#### 4.1: Update Existing Feature Sets (Week 5)

**Changes**:
- Update `HalfKP`, `HalfKA`, etc. if you want to keep them for comparison
- Replace king-based features with den-based or piece-count-based features

**Example - HalfKP Adaptation**:
```cpp
// Original: King-Piece features
struct HalfKP {
    static constexpr int NUM_SQ = 64;
    static constexpr int NUM_PT = 10;  // Excludes kings
    static constexpr int INPUTS = NUM_SQ * (NUM_SQ * NUM_PT + 1);
};

// Jungle: Could use "piece-perspective" instead of king-perspective
struct HalfPP { // Piece-Piece features (if no king)
    static constexpr int NUM_SQ = 63;
    static constexpr int NUM_PT = 16;  // All piece-color combinations
    // Or: Anchor on strongest piece, or use global features
};
```

**Decision Point**:
- Do you need king-equivalent features? (Den location is fixed, not useful as anchor)
- Consider: Anchor on strongest piece position, or use position-independent features

**Validation**:
- [ ] Feature extraction compiles
- [ ] Correct feature count for Jungle board
- [ ] Unit test: Feature indices are valid (< INPUTS)

**Commit**: `Phase 4.1: Update existing feature sets for Jungle board geometry`

#### 4.2: Implement HalfATA Feature Set (Week 5-6)

**Goal**: Complete the custom HalfATA feature set (lines 370-459 in training_data_loader.cpp)

**Design** (based on your stubs):
```cpp
struct HalfATA {
    static constexpr int NUM_SQ = 63;
    static constexpr int NUM_PT = 16;  // 8 piece types × 2 colors
    static constexpr int NUM_ATTACK_BUCKETS = 6;  // Define attack categories
    static constexpr int NUM_PLANES = NUM_SQ * NUM_PT;
    static constexpr int INPUTS = NUM_PLANES * NUM_ATTACK_BUCKETS;

    // Attack buckets:
    // 0: Losing (opponent has advantage)
    // 1: Slightly losing
    // 2: Equal
    // 3: Slightly winning
    // 4: Winning (player has advantage)
    // 5: Dominating (near-win position)

    static int classify_attack_buckets(const Position& pos) {
        // TODO: Implement attack evaluation
        // Possible metrics:
        // - Material count weighted by piece power
        // - Control of dens/traps
        // - Piece mobility
        // - Threat to opponent pieces

        int material_balance = calculate_material(pos);
        int positional_score = evaluate_position(pos);

        int total = material_balance + positional_score;

        if (total < -300) return 0;
        if (total < -100) return 1;
        if (total < 100) return 2;
        if (total < 300) return 3;
        if (total < 500) return 4;
        return 5;
    }

    static std::pair<int, int> fill_features_sparse(...) {
        // Already stubbed at line 412
        // Update to use attack bucket from classify_attack_buckets
    }
};
```

**Sub-tasks**:
- [ ] Define attack bucket classification logic
- [ ] Implement material evaluation for Jungle pieces
- [ ] Implement positional evaluation (den control, trap usage, mobility)
- [ ] Complete `fill_features_sparse` implementation
- [ ] Implement `HalfATAFactorized` variant

**Validation**:
- [ ] Unit test: Attack bucket classification for known positions
- [ ] Unit test: Feature extraction produces valid indices
- [ ] Integration test: Create SparseBatch with HalfATA features
- [ ] Verify feature values are reasonable (mostly 1.0 for sparse)

**Commit**: `Phase 4.2: Implement HalfATA feature set for Jungle chess`

#### 4.3: Register New Feature Sets (Week 6)

**Changes**:
- Update `get_sparse_batch_from_fens` (line 1121)
- Update `create_sparse_batch_stream` (line 1204)
- Add feature set names: "HalfATA", "HalfATA^"

**Validation**:
- [ ] C API accepts "HalfATA" feature set name
- [ ] SparseBatch created successfully with Jungle positions
- [ ] Batch contains correct dimensions and feature counts

**Commit**: `Phase 4.3: Register HalfATA in C API and validate`

---

### Phase 5: End-to-End Integration Testing
**Goal**: Validate entire pipeline with real Jungle training data

**Estimated Scope**: Testing and debugging

#### 5.1: Create Test Dataset (Week 6)

**Tasks**:
- [ ] Generate or collect 1000+ Jungle chess games
- [ ] Convert to .binpack format using updated writer
- [ ] Create small test set (100 positions) and large test set (100K positions)

**Validation**:
- [ ] Files readable by data loader
- [ ] No crashes on edge cases (start position, endgame, etc.)

#### 5.2: Performance Benchmarking (Week 7)

**Tasks**:
- [ ] Run `training_data_loader_benchmark` with Jungle data
- [ ] Compare throughput with chess data (MPos/s)
- [ ] Profile for bottlenecks (use PGO build)

**Validation**:
- [ ] Performance within acceptable range (>1 MPos/s for HalfATA)
- [ ] No memory leaks (valgrind or similar)
- [ ] Multi-threading scales properly

**Commit**: `Phase 5.2: Performance validation and optimization`

#### 5.3: Python Integration (Week 7)

**Tasks**:
- [ ] Update Python `nnue_dataset.py` (not in repo) to match new feature sets
- [ ] Test ctypes bindings with new C API
- [ ] Create end-to-end training smoke test (1 epoch, small model)

**Validation**:
- [ ] Python successfully loads batches
- [ ] SparseBatch tensors have correct shapes
- [ ] Feature indices are within bounds
- [ ] Training loop runs without crashes

**Commit**: `Phase 5.3: Validate Python integration and training pipeline`

---

### Phase 6: Documentation and Cleanup
**Goal**: Production-ready code with documentation

#### 6.1: Update Documentation (Week 7)

**Tasks**:
- [ ] Update CLAUDE.md for Jungle chess
- [ ] Document new feature sets (HalfATA)
- [ ] Add Jungle-specific notes (board layout, piece rules)
- [ ] Update build instructions if needed

#### 6.2: Code Cleanup (Week 7-8)

**Tasks**:
- [ ] Remove unused chess-specific code (if any remains)
- [ ] Update comments (translate relevant Chinese comments if needed)
- [ ] Add Doxygen/comments for new Jungle-specific functions
- [ ] Run linter/formatter

#### 6.3: Final Testing (Week 8)

**Tasks**:
- [ ] Clean build from scratch
- [ ] Test on different platforms (Linux, macOS if applicable)
- [ ] Regression test suite (automated tests for all phases)
- [ ] Load testing with large datasets (1M+ positions)

**Commit**: `Phase 6: Production-ready Jungle chess data loader`

---

## Risk Management

### High-Risk Areas

1. **Bitboard Assumptions**:
   - Risk: 63 squares don't fit cleanly in 64-bit bitboards
   - Mitigation: Use 64-bit with one unused bit, or switch to different representation

2. **Move Generation Complexity**:
   - Risk: Jungle rules (river jumps, traps) are complex
   - Mitigation: Start with simplified rules, add edge cases incrementally

3. **Feature Set Compatibility**:
   - Risk: Existing training code assumes chess feature dimensions
   - Mitigation: Coordinate with Python side early (Phase 5.3)

4. **Binary Format Changes**:
   - Risk: Existing .binpack tools may break
   - Mitigation: Keep chess reader in separate branch, or version the format

### Rollback Strategy

Each phase has a git commit. If a phase fails:
1. Identify failing commit with `git bisect`
2. Revert to last working commit
3. Create minimal reproducer
4. Fix incrementally with smaller commits

---

## Testing Strategy

### Unit Tests (Each Phase)
```cpp
// Example test structure
namespace test {
    void test_square_enumeration();
    void test_piece_movement();
    void test_capture_rules();
    void test_feature_extraction();
    // ... ~20-30 unit tests total
}
```

### Integration Tests (Phase 5)
- End-to-end: Load Jungle game → Extract features → Create batch
- Stress test: Process 1M positions without crash
- Correctness: Compare hand-calculated features vs. code output

### Continuous Validation
After each commit:
```bash
make clean && sh compile_data_loader.bat
./run_tests  # Unit tests
./build/training_data_loader_benchmark test_data/jungle_sample.binpack
```

---

## Timeline Estimate

| Phase | Duration | Effort | Risk |
|-------|----------|--------|------|
| 0. Setup | 2-3 days | Low | Low |
| 1. Data Structures | 1-2 weeks | Medium | Medium |
| 2. Move Generation | 1-2 weeks | High | High |
| 3. Binary Format | 1 week | Medium | Medium |
| 4. Feature Sets | 1-2 weeks | Medium | Medium |
| 5. Integration | 1 week | Medium | Low |
| 6. Documentation | 3-5 days | Low | Low |
| **Total** | **6-8 weeks** | | |

**Assumptions**:
- 1 developer, part-time (~20 hrs/week)
- Access to Jungle chess rules reference
- Existing test data or ability to generate it

---

## Success Criteria

- ✅ All code compiles without warnings
- ✅ Unit tests pass (>90% coverage of new code)
- ✅ Benchmark achieves >1 MPos/s with HalfATA features
- ✅ Python training loop runs successfully
- ✅ Documentation complete and accurate
- ✅ No memory leaks or crashes in stress tests

---

## Open Questions (Resolve Before Starting)

1. **Feature Set Design**: Which feature set is primary? HalfATA only, or also adapt HalfKA/HalfKP?
2. **Training Data Source**: Do you have existing Jungle chess .binpack files, or need to generate?
3. **Python Side**: Who maintains the Python training code? Need coordination?
4. **Bitboard Representation**: 64-bit with 1 unused bit OK, or need custom 63-bit solution?
5. **Terrain Encoding**: Should special squares (traps, dens, rivers) be explicit in Position, or implicit in move rules?

---

## Next Steps

1. **Review this plan** with stakeholders
2. **Answer open questions** (esp. #1, #2, #5)
3. **Set up Phase 0** (branch, test data, baseline)
4. **Begin Phase 1.1** (Square geometry adaptation)

---

## Appendix: Key File Modification Summary

| File | Lines | Main Changes |
|------|-------|--------------|
| `lib/nnue_training_data_formats.h` | ~8200 | Chess→Jungle: Square(64→63), Pieces(6→8), Move types, Position logic |
| `training_data_loader.cpp` | ~1400 | Feature sets: Update dimensions, implement HalfATA, remove king-based anchors |
| `lib/nnue_training_data_stream.h` | ~260 | Minimal (format-agnostic), verify compatibility |
| `CMakeLists.txt` | ~82 | Possibly update for new namespaces/tests |
| `lib/rng.h` | ~18 | No changes needed |

**Total estimated changes**: ~1500-2500 lines modified/added across all files.
