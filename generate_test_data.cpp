/**
 * Generate a minimal test .bin file with a few chess positions
 * This creates a small dataset for testing the data loader
 */
#include <fstream>
#include <iostream>
#include <vector>
#include "lib/nnue_training_data_formats.h"

using namespace chess;
using namespace binpack;

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " output.bin\n";
        return 1;
    }

    const char* output_file = argv[1];
    std::ofstream outfile(output_file, std::ios::binary);

    if (!outfile) {
        std::cerr << "Failed to open output file: " << output_file << "\n";
        return 1;
    }

    // Create a vector of test positions
    std::vector<std::string> test_fens = {
        // Starting position
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        // After e4
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        // After e4 e5
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
        // After e4 e5 Nf3
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
        // After e4 e5 Nf3 Nc6
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        // Italian game
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        // Sicilian
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
        // French defense
        "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        // Caro-Kann
        "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        // Queen's Gambit
        "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2",
        // King's Indian setup
        "rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 1 3",
        // Scandinavian defense
        "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
        // English opening
        "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq c3 0 1",
        // Reti opening
        "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1",
        // Bird's opening
        "rnbqkbnr/pppppppp/8/8/5P2/8/PPPPP1PP/RNBQKBNR b KQkq f3 0 1",
        // Middlegame position
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5",
        // Endgame - KR vs K
        "8/8/8/8/8/4k3/8/4K2R w - - 0 1",
        // Endgame - KQ vs K
        "8/8/8/8/8/4k3/8/4KQ2 w - - 0 1",
        // Endgame - KP vs K
        "8/8/8/8/8/4k3/4P3/4K3 w - - 0 1",
        // Complex middlegame
        "r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 8 9"
    };

    std::cout << "Generating test file with " << test_fens.size() << " positions...\n";

    int count = 0;
    for (const auto& fen : test_fens) {
        // Parse FEN to Position
        Position pos = Position::fromFen(fen);

        // Create a TrainingDataEntry
        TrainingDataEntry entry;
        entry.pos = pos;

        // Generate a plausible move (just use a null move as dummy)
        // In real data this would be the best move from the position
        entry.move = Move::null();

        // Add some plausible values
        entry.score = static_cast<int16_t>((count * 17) % 200 - 100); // Scores from -100 to +100
        entry.ply = static_cast<uint16_t>(count % 80); // Ply from 0-79
        entry.result = static_cast<int16_t>((count % 3) - 1); // -1, 0, or 1

        // Convert to packed format and write
        nodchip::PackedSfenValue packed = trainingDataEntryToPackedSfenValue(entry);
        outfile.write(reinterpret_cast<const char*>(&packed), sizeof(nodchip::PackedSfenValue));

        count++;
    }

    outfile.close();

    std::cout << "Successfully wrote " << count << " positions to " << output_file << "\n";
    std::cout << "File size: " << (count * sizeof(nodchip::PackedSfenValue)) << " bytes\n";

    return 0;
}
