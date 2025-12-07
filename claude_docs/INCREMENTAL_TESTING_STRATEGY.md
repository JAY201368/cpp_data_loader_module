# Incremental Testing Strategy: Dual-Mode Development

## Core Strategy: Namespace Isolation + Compile-Time Selection

The key insight is to **run both chess and Jungle side-by-side** during migration, allowing you to:
- ✅ Validate each Jungle change against working chess reference
- ✅ Run existing chess tests to ensure no regression
- ✅ Incrementally test Jungle components as they're built
- ✅ Compare outputs (feature extraction, move generation) between systems

---

## Architecture: Three-Namespace Design

```
lib/nnue_training_data_formats.h:
├── namespace chess { ... }      // Original, unchanged
├── namespace jungle { ... }     // New, incrementally built
└── namespace common { ... }     // Shared utilities (bitboards, etc.)

training_data_loader.cpp:
├── Feature sets for chess (existing)
├── Feature sets for jungle (new)
└── Dual C API (chess_*, jungle_*)
```

---

## Phase-by-Phase Testing Strategy

### Phase 0: Testing Infrastructure Setup

#### Step 0.1: Create Test Harness (Day 1-2)

Create `test/test_runner.cpp`:

```cpp
#include "../lib/nnue_training_data_formats.h"
#include "../training_data_loader.cpp"
#include <cassert>
#include <iostream>

// Test fixture that can run both chess and jungle tests
namespace test {
    int tests_run = 0;
    int tests_passed = 0;

    #define TEST(name) \
        void test_##name(); \
        void run_##name() { \
            std::cout << "Running " #name "..." << std::flush; \
            tests_run++; \
            test_##name(); \
            tests_passed++; \
            std::cout << " PASSED\n"; \
        } \
        void test_##name()

    #define ASSERT_EQ(a, b) \
        if ((a) != (b)) { \
            std::cerr << "\nAssertion failed: " #a " == " #b \
                      << "\n  Expected: " << (b) \
                      << "\n  Got: " << (a) << "\n"; \
            std::abort(); \
        }

    #define ASSERT_TRUE(cond) \
        if (!(cond)) { \
            std::cerr << "\nAssertion failed: " #cond << "\n"; \
            std::abort(); \
        }

    // Chess baseline tests
    TEST(chess_square_count) {
        ASSERT_EQ(chess::Square::Count, 64);
    }

    TEST(chess_piece_types) {
        using chess::PieceType;
        ASSERT_EQ(static_cast<int>(PieceType::King), 6);
    }

    TEST(chess_move_generation) {
        // Test starting position has 20 legal moves
        auto pos = chess::Position::startingPosition();
        int move_count = 0;
        chess::movegen::forEachLegalMove(pos, [&](chess::Move m) {
            move_count++;
        });
        ASSERT_EQ(move_count, 20);
    }

    // Jungle stub tests (initially empty/skipped)
    TEST(jungle_square_count) {
        #ifdef JUNGLE_ENABLED
        ASSERT_EQ(jungle::Square::Count, 63);
        #else
        std::cout << " SKIPPED (JUNGLE_ENABLED not defined)";
        #endif
    }

    void run_all() {
        std::cout << "=== Running Chess Tests ===\n";
        run_chess_square_count();
        run_chess_piece_types();
        run_chess_move_generation();

        std::cout << "\n=== Running Jungle Tests ===\n";
        run_jungle_square_count();

        std::cout << "\n=== Summary ===\n";
        std::cout << tests_passed << "/" << tests_run << " tests passed\n";
        if (tests_passed != tests_run) {
            exit(1);
        }
    }
}

int main() {
    test::run_all();
    return 0;
}
```

#### Step 0.2: Integrate into Build System

Update `CMakeLists.txt`:

```cmake
# Add at the end
option(BUILD_TESTS "Build test suite" ON)
option(ENABLE_JUNGLE "Enable Jungle chess support" OFF)

if(ENABLE_JUNGLE)
    add_definitions(-DJUNGLE_ENABLED)
endif()

if(BUILD_TESTS)
    add_executable(test_runner test/test_runner.cpp)
    target_link_libraries(test_runner Threads::Threads)

    # Add test target
    enable_testing()
    add_test(NAME unit_tests COMMAND test_runner)
endif()
```

#### Validation Loop:

```bash
# Build and run chess tests (baseline)
cmake -S . -B build -DBUILD_TESTS=ON -DENABLE_JUNGLE=OFF
cmake --build build
./build/test_runner

# Expected output:
# === Running Chess Tests ===
# Running chess_square_count... PASSED
# Running chess_piece_types... PASSED
# Running chess_move_generation... PASSED
# === Running Jungle Tests ===
# Running jungle_square_count... SKIPPED (JUNGLE_ENABLED not defined)
# 3/4 tests passed
```

**Commit**: `Phase 0: Add test harness with chess baseline tests`

---

### Phase 1: Core Data Structures (With Continuous Validation)

#### Phase 1.1: Add Jungle Namespace Skeleton (Day 1)

In `lib/nnue_training_data_formats.h`, **after** chess namespace:

```cpp
// Around line 6310, before binpack namespace
namespace jungle {
    // Initially, just copy-paste chess namespace structure
    // with TODOs for what needs changing

    using FileEnum = chess::FileEnum;  // TEMPORARY: reuse chess
    using RankEnum = chess::RankEnum;  // TEMPORARY: reuse chess
    using Square = chess::Square;      // TEMPORARY: reuse chess

    enum class PieceType : uint8_t {
        None, Rat, Cat, Dog, Wolf, Leopard, Tiger, Lion, Elephant
    };

    // TODO: Later phases will diverge from chess
    using Position = chess::Position;  // TEMPORARY
    using Move = chess::Move;          // TEMPORARY
}
```

**Test Update**:

```cpp
TEST(jungle_namespace_exists) {
    #ifdef JUNGLE_ENABLED
    // Just test that jungle namespace compiles
    jungle::PieceType rat = jungle::PieceType::Rat;
    ASSERT_EQ(static_cast<int>(rat), 1);
    #else
    std::cout << " SKIPPED";
    #endif
}
```

**Validation**:

```bash
# Test with Jungle enabled
cmake -S . -B build -DENABLE_JUNGLE=ON
cmake --build build
./build/test_runner

# Both chess and jungle tests should pass
# === Running Chess Tests ===
# Running chess_square_count... PASSED
# ... (all chess tests pass)
# === Running Jungle Tests ===
# Running jungle_namespace_exists... PASSED
# 4/4 tests passed
```

**Commit**: `Phase 1.1: Add jungle namespace skeleton (reuses chess for now)`

---

#### Phase 1.2: Implement Jungle Square (7×9) (Day 2-3)

Replace temporary jungle::Square with real implementation:

```cpp
namespace jungle {
    enum FileEnum : uint8_t {
        fileA, fileB, fileC, fileD, fileE, fileF, fileG, fileNone
    };

    enum RankEnum : uint8_t {
        rank1, rank2, rank3, rank4, rank5, rank6, rank7, rank8, rank9, rankNone
    };

    struct Square {
        static constexpr int Count = 63;  // 7×9

        uint8_t m_value;

        constexpr Square() : m_value(64) {}
        constexpr Square(FileEnum file, RankEnum rank)
            : m_value(static_cast<uint8_t>(file) + static_cast<uint8_t>(rank) * 7) {}

        // ... implement all methods (similar to chess::Square)

        [[nodiscard]] constexpr FileEnum file() const {
            return static_cast<FileEnum>(m_value % 7);
        }

        [[nodiscard]] constexpr RankEnum rank() const {
            return static_cast<RankEnum>(m_value / 7);
        }

        [[nodiscard]] constexpr bool isValid() const {
            return m_value < 63;
        }
    };
}
```

**Test Update**:

```cpp
TEST(jungle_square_count) {
    #ifdef JUNGLE_ENABLED
    ASSERT_EQ(jungle::Square::Count, 63);
    #else
    std::cout << " SKIPPED";
    #endif
}

TEST(jungle_square_arithmetic) {
    #ifdef JUNGLE_ENABLED
    jungle::Square a1(jungle::fileA, jungle::rank1);
    jungle::Square g9(jungle::fileG, jungle::rank9);

    ASSERT_EQ(a1.file(), jungle::fileA);
    ASSERT_EQ(a1.rank(), jungle::rank1);
    ASSERT_EQ(g9.file(), jungle::fileG);
    ASSERT_EQ(g9.rank(), jungle::rank9);

    // Test all 63 squares are valid
    for (int i = 0; i < 63; i++) {
        jungle::Square sq;
        sq.m_value = i;
        ASSERT_TRUE(sq.isValid());
    }
    #else
    std::cout << " SKIPPED";
    #endif
}

TEST(jungle_square_vs_chess) {
    // Comparison test: chess has 64, jungle has 63
    ASSERT_EQ(chess::Square::Count, 64);
    #ifdef JUNGLE_ENABLED
    ASSERT_EQ(jungle::Square::Count, 63);
    #endif
}
```

**Validation**:

```bash
cmake -S . -B build -DENABLE_JUNGLE=ON
cmake --build build
./build/test_runner

# Expected:
# === Running Jungle Tests ===
# Running jungle_square_count... PASSED
# Running jungle_square_arithmetic... PASSED
# Running jungle_square_vs_chess... PASSED
```

**Critical**: At this point, chess tests still pass (unchanged namespace), and Jungle square system is validated.

**Commit**: `Phase 1.2: Implement jungle::Square for 7×9 board`

---

#### Phase 1.3: Implement Jungle Pieces (Day 3-4)

```cpp
namespace jungle {
    enum class PieceType : uint8_t {
        None, Rat, Cat, Dog, Wolf, Leopard, Tiger, Lion, Elephant
    };

    struct Piece {
        uint8_t m_value;

        // Similar to chess::Piece but with jungle piece types
        constexpr Piece(PieceType type, Color color)
            : m_value((static_cast<uint8_t>(type) << 1) | static_cast<uint8_t>(color)) {}

        [[nodiscard]] constexpr PieceType type() const {
            return static_cast<PieceType>(m_value >> 1);
        }

        [[nodiscard]] constexpr Color color() const {
            return static_cast<Color>(m_value & 1);
        }

        [[nodiscard]] constexpr int power() const {
            // Jungle-specific: piece power for capture rules
            return static_cast<int>(type());  // Rat=1, Cat=2, ..., Elephant=8
        }
    };
}
```

**Test Update**:

```cpp
TEST(jungle_piece_types) {
    #ifdef JUNGLE_ENABLED
    using jungle::PieceType;
    using jungle::Piece;
    using jungle::Color;

    // Test all 8 piece types
    Piece rat(PieceType::Rat, Color::White);
    Piece elephant(PieceType::Elephant, Color::Black);

    ASSERT_EQ(rat.type(), PieceType::Rat);
    ASSERT_EQ(rat.power(), 1);
    ASSERT_EQ(elephant.type(), PieceType::Elephant);
    ASSERT_EQ(elephant.power(), 8);

    // Test 16 piece-color combinations
    int count = 0;
    for (int t = 1; t <= 8; t++) {
        for (int c = 0; c < 2; c++) {
            Piece p(static_cast<PieceType>(t), static_cast<Color>(c));
            ASSERT_TRUE(p.type() != PieceType::None);
            count++;
        }
    }
    ASSERT_EQ(count, 16);
    #else
    std::cout << " SKIPPED";
    #endif
}

TEST(jungle_capture_rules) {
    #ifdef JUNGLE_ENABLED
    using jungle::PieceType;
    using jungle::Piece;
    using jungle::Color;

    // Test power hierarchy
    Piece elephant(PieceType::Elephant, Color::White);
    Piece lion(PieceType::Lion, Color::Black);
    Piece rat(PieceType::Rat, Color::White);

    ASSERT_TRUE(elephant.power() > lion.power());  // 8 > 7
    ASSERT_TRUE(lion.power() > rat.power());       // 7 > 1

    // Special rule: Rat can capture Elephant (implemented later)
    #else
    std::cout << " SKIPPED";
    #endif
}
```

**Validation**: Same as before - both chess and jungle tests pass.

**Commit**: `Phase 1.3: Implement jungle::Piece with 8 piece types`

---

### Phase 2: Move Generation (With Dual Testing)

#### Phase 2.1: Jungle Position Skeleton (Day 5-6)

Strategy: Create `jungle::Position` that initially just wraps a 7×9 board, no move logic yet.

```cpp
namespace jungle {
    struct Position {
        // Simplified: just piece placement for now
        std::array<Piece, 63> m_pieces;
        Color m_sideToMove;

        Position() {
            for (auto& p : m_pieces) {
                p = Piece(PieceType::None, Color::White);
            }
            m_sideToMove = Color::White;
        }

        static Position startingPosition() {
            Position pos;
            // TODO: Set up Jungle starting position
            // For now, just place a few pieces for testing
            pos.m_pieces[0] = Piece(PieceType::Lion, Color::White);   // A1
            pos.m_pieces[62] = Piece(PieceType::Lion, Color::Black);  // G9
            return pos;
        }

        [[nodiscard]] Piece pieceAt(Square sq) const {
            return m_pieces[sq.m_value];
        }

        void setPiece(Square sq, Piece piece) {
            m_pieces[sq.m_value] = piece;
        }
    };
}
```

**Test Update**:

```cpp
TEST(jungle_starting_position) {
    #ifdef JUNGLE_ENABLED
    auto pos = jungle::Position::startingPosition();

    // Test a few known pieces
    jungle::Square a1(jungle::fileA, jungle::rank1);
    jungle::Square g9(jungle::fileG, jungle::rank9);

    auto piece_a1 = pos.pieceAt(a1);
    auto piece_g9 = pos.pieceAt(g9);

    ASSERT_EQ(piece_a1.type(), jungle::PieceType::Lion);
    ASSERT_EQ(piece_g9.type(), jungle::PieceType::Lion);
    ASSERT_EQ(piece_a1.color(), jungle::Color::White);
    ASSERT_EQ(piece_g9.color(), jungle::Color::Black);
    #else
    std::cout << " SKIPPED";
    #endif
}

TEST(jungle_position_manipulation) {
    #ifdef JUNGLE_ENABLED
    jungle::Position pos;
    jungle::Square d5(jungle::fileD, jungle::rank5);

    // Place a piece
    jungle::Piece rat(jungle::PieceType::Rat, jungle::Color::White);
    pos.setPiece(d5, rat);

    // Verify
    auto retrieved = pos.pieceAt(d5);
    ASSERT_EQ(retrieved.type(), jungle::PieceType::Rat);
    #else
    std::cout << " SKIPPED";
    #endif
}
```

**Validation**: Position data structure works, no move generation yet.

**Commit**: `Phase 2.1: Implement jungle::Position with basic piece placement`

---

#### Phase 2.2: Implement Move Generation (Day 7-10)

Now implement actual Jungle move rules. **Key testing strategy**: Compare move counts with hand-calculated values.

```cpp
namespace jungle {
    namespace movegen {
        template <typename F>
        void forEachPseudoLegalMove(const Position& pos, F&& callback) {
            // Implement Jungle move rules
            // Start with simple orthogonal moves, add special rules later

            for (int i = 0; i < 63; i++) {
                Square from(i);
                Piece piece = pos.pieceAt(from);

                if (piece.type() == PieceType::None) continue;
                if (piece.color() != pos.sideToMove()) continue;

                // Try all 4 orthogonal directions
                // TODO: Add river jump for Lion/Tiger
                // TODO: Add river entry for Rat
            }
        }
    }
}
```

**Test Update**:

```cpp
TEST(jungle_move_generation_basic) {
    #ifdef JUNGLE_ENABLED
    // Set up a simple position: one piece in center
    jungle::Position pos;
    jungle::Square d5(jungle::fileD, jungle::rank5);
    pos.setPiece(d5, jungle::Piece(jungle::PieceType::Dog, jungle::Color::White));
    pos.m_sideToMove = jungle::Color::White;

    // Count moves (should be 4: up, down, left, right)
    int move_count = 0;
    jungle::movegen::forEachPseudoLegalMove(pos, [&](jungle::Move m) {
        move_count++;
    });

    ASSERT_EQ(move_count, 4);
    #else
    std::cout << " SKIPPED";
    #endif
}

TEST(jungle_lion_river_jump) {
    #ifdef JUNGLE_ENABLED
    // Test Lion can jump over river
    jungle::Position pos;
    // Place Lion at edge of river
    // ... setup specific scenario

    int jump_moves = 0;
    jungle::movegen::forEachPseudoLegalMove(pos, [&](jungle::Move m) {
        // Count only river jump moves
        if (/* move crosses river */) jump_moves++;
    });

    ASSERT_TRUE(jump_moves > 0);
    #else
    std::cout << " SKIPPED";
    #endif
}

// Comparison test: verify chess still works
TEST(chess_move_generation_regression) {
    auto pos = chess::Position::startingPosition();
    int move_count = 0;
    chess::movegen::forEachLegalMove(pos, [&](chess::Move m) {
        move_count++;
    });
    ASSERT_EQ(move_count, 20);  // Still 20 moves in chess starting position
}
```

**Validation Loop**:

```bash
# After each move rule addition:
cmake --build build
./build/test_runner

# All tests should pass:
# - Chess tests unchanged (regression check)
# - Jungle tests validate new move rules
```

**Incremental Commits**:
- `Phase 2.2a: Basic orthogonal movement for all pieces`
- `Phase 2.2b: Add Lion/Tiger river jump`
- `Phase 2.2c: Add Rat river movement`
- `Phase 2.2d: Add capture rules with power hierarchy`

---

### Phase 3: Feature Extraction (Side-by-Side Testing)

#### Phase 3.1: Duplicate Feature Sets for Jungle

Create parallel feature sets:

```cpp
// In training_data_loader.cpp

// Existing chess feature set (unchanged)
struct ChessHalfKP {
    static constexpr int NUM_SQ = 64;
    static constexpr int NUM_PT = 10;
    // ... existing implementation
};

// New Jungle feature set
struct JungleHalfATA {
    static constexpr int NUM_SQ = 63;
    static constexpr int NUM_PT = 16;
    static constexpr int NUM_ATTACK_BUCKETS = 6;
    static constexpr int INPUTS = NUM_SQ * NUM_PT * NUM_ATTACK_BUCKETS;

    static std::pair<int, int> fill_features_sparse(
        const jungle::TrainingDataEntry& e,  // Note: jungle namespace
        int* features,
        float* values,
        jungle::Color color
    ) {
        // Implement feature extraction for Jungle
    }
};
```

**Test Update**:

```cpp
TEST(jungle_feature_extraction) {
    #ifdef JUNGLE_ENABLED
    jungle::Position pos = jungle::Position::startingPosition();
    jungle::TrainingDataEntry entry;
    entry.pos = pos;

    int features[100];
    float values[100];

    auto [num_features, feature_dim] = JungleHalfATA::fill_features_sparse(
        entry, features, values, jungle::Color::White
    );

    // Verify all feature indices are valid
    for (int i = 0; i < num_features; i++) {
        ASSERT_TRUE(features[i] >= 0);
        ASSERT_TRUE(features[i] < JungleHalfATA::INPUTS);
        ASSERT_EQ(values[i], 1.0f);  // Sparse features should be 1.0
    }

    ASSERT_EQ(feature_dim, JungleHalfATA::INPUTS);
    #else
    std::cout << " SKIPPED";
    #endif
}

// Comparison test
TEST(chess_feature_extraction_regression) {
    chess::Position pos = chess::Position::startingPosition();
    chess::TrainingDataEntry entry;
    entry.pos = pos;

    int features[100];
    float values[100];

    auto [num_features, feature_dim] = ChessHalfKP::fill_features_sparse(
        entry, features, values, chess::Color::White
    );

    ASSERT_TRUE(num_features > 0);
    ASSERT_EQ(feature_dim, ChessHalfKP::INPUTS);
}
```

**Validation**:

```bash
./build/test_runner

# Both systems work:
# === Running Chess Tests ===
# ... (all chess tests pass)
# === Running Jungle Tests ===
# ... (all jungle tests pass)
```

**Commit**: `Phase 3.1: Add JungleHalfATA feature set with side-by-side testing`

---

### Phase 4: End-to-End Integration Test

#### Phase 4.1: Create Dual C API

```cpp
extern "C" {
    // Chess API (existing, unchanged)
    EXPORT SparseBatch* create_chess_sparse_batch_stream(...) {
        // Existing implementation using chess::
    }

    // Jungle API (new)
    EXPORT SparseBatch* create_jungle_sparse_batch_stream(
        const char* feature_set_c,
        int concurrency,
        int num_files,
        const char* const* filenames,
        int batch_size,
        bool cyclic
    ) {
        #ifdef JUNGLE_ENABLED
        std::string_view feature_set(feature_set_c);
        if (feature_set == "JungleHalfATA") {
            return new FeaturedBatchStream<
                FeatureSet<JungleHalfATA>,
                SparseBatch
            >(concurrency, filenames_vec, batch_size, cyclic, nullptr);
        }
        #endif
        return nullptr;
    }
}
```

#### Phase 4.2: End-to-End Test

```cpp
TEST(jungle_end_to_end) {
    #ifdef JUNGLE_ENABLED
    // 1. Create a test .binpack file with Jungle positions
    // 2. Load it through data loader
    // 3. Verify batches are correct

    const char* test_file = "test_data/jungle_sample.binpack";
    const char* files[] = {test_file};

    auto stream = create_jungle_sparse_batch_stream(
        "JungleHalfATA", 1, 1, files, 8, false
    );

    ASSERT_TRUE(stream != nullptr);

    auto batch = fetch_next_sparse_batch(stream);
    ASSERT_TRUE(batch != nullptr);

    // Verify batch properties
    ASSERT_EQ(batch->size, 8);
    ASSERT_EQ(batch->num_inputs, JungleHalfATA::INPUTS);

    destroy_sparse_batch(batch);
    destroy_sparse_batch_stream(stream);
    #else
    std::cout << " SKIPPED";
    #endif
}

TEST(chess_end_to_end_regression) {
    // Same test for chess - ensure it still works
    const char* test_file = "test_data/chess_sample.binpack";
    // ... similar test
}
```

**Validation**:

```bash
# Run with both chess and jungle enabled
cmake -S . -B build -DENABLE_JUNGLE=ON -DBUILD_TESTS=ON
cmake --build build
./build/test_runner

# Create test data
./build/test_runner --generate-test-data

# Run full suite
./build/test_runner

# === Summary ===
# Chess tests: 15/15 passed
# Jungle tests: 20/20 passed (when JUNGLE_ENABLED)
# Total: 35/35 passed
```

---

## Summary: Testing Strategy Benefits

### ✅ **Continuous Validation**
- Chess tests run **every build** → catch regressions immediately
- Jungle tests run incrementally → validate each component as built
- No "big bang" integration → errors isolated to recent changes

### ✅ **Side-by-Side Comparison**
- Feature extraction: Compare chess vs jungle outputs
- Move generation: Validate move counts against known values
- Performance: Benchmark both systems to catch slowdowns

### ✅ **Rollback Safety**
- Each commit has passing tests
- Git bisect works: `git bisect run ./build/test_runner`
- Can revert to any working state

### ✅ **Incremental Complexity**
- Phase 1: Data structures only (pure functions, easy to test)
- Phase 2: Move generation (testable with known positions)
- Phase 3: Feature extraction (testable with unit tests)
- Phase 4: End-to-end (integration tests)

---

## Build Commands Cheat Sheet

```bash
# Chess only (baseline)
cmake -S . -B build -DENABLE_JUNGLE=OFF -DBUILD_TESTS=ON
cmake --build build && ./build/test_runner

# Chess + Jungle (development)
cmake -S . -B build -DENABLE_JUNGLE=ON -DBUILD_TESTS=ON
cmake --build build && ./build/test_runner

# Production build (Jungle only, no tests)
cmake -S . -B build -DENABLE_JUNGLE=ON -DBUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Run specific test
./build/test_runner --filter=jungle_move_generation

# Continuous testing during development
watch -n 2 "cmake --build build && ./build/test_runner"
```

---

## Next Steps

1. **Implement Phase 0**: Set up test harness
2. **Add first Jungle test**: `jungle_namespace_exists`
3. **Validate baseline**: All chess tests pass
4. **Begin Phase 1.1**: Add Jungle namespace skeleton
5. **Iterate**: Add one component → test → commit → repeat

Would you like me to help you implement the test harness code (Phase 0)?
