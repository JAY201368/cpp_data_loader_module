/**
 * 处理加载训练数据(.bin, .binpack)并为批处理准备批次
 * 由于性能要求, 必须以本地方式实现训练数据加载器. 数据加载器可以使用 compile_data_loader.bat 编译, 该脚本使用 cmake. 如果知道如何操作, 也可以在没有 cmake 的情况下进行编译. 
 * 该组件实现了将棋盘状态分解为单个特征. 此步骤必须与在 Python 端定义的特征集一致. 
 * 数据加载器一次提供整个批次, 以消除稍后将单个样本连接成批次的成本. SparseBatch 表示这样的批次. 它为样本的每个属性分配一个数组. 
 * 活跃特征以(活跃特征索引, 活跃特征值)pair的形式存储. Pos对象中活跃特征的上限已知. 未使用的特征以索引 -1 存储. 
 * 数据加载器支持多线程. 线程的数量可以在创建时指定, 并在读取磁盘数据的线程和形成批次的线程之间进行启发式分配. 
 * 数据加载器还支持形成一个 fen 字符串数组, 而不是训练批次. 这对于其他工具非常有用, 例如 cross_check_eval.py. 
 * 绑定是通过 nnue_dataset.py 组件中的 ctypes 完成的. 
 */
#include <iostream>
#include <memory>
#include <string>
#include <algorithm>
#include <iterator>
#include <future>
#include <mutex>
#include <thread>
#include <deque>
#include <random>

#include "lib/nnue_training_data_formats.h"
#include "lib/nnue_training_data_stream.h"
#include "lib/rng.h"

#if defined (__x86_64__)
#define EXPORT
#define CDECL
#else
#if defined (_MSC_VER)
#define EXPORT __declspec(dllexport)
#define CDECL __cdecl
#else
#define EXPORT
#define CDECL __attribute__ ((__cdecl__))
#endif
#endif

using namespace binpack;
using namespace chess;

static Square orient(Color color, Square sq)
{
    if (color == Color::White)
    {
        return sq;
    }
    else
    {
        // IMPORTANT: for now we use rotate180 instead of rank flip
        //            for compatibility with the stockfish master branch.
        //            Note that this is inconsistent with nodchip/master.
        return sq.flippedVertically().flippedHorizontally();
    }
}

static Square orient_flip(Color color, Square sq)
{
    if (color == Color::White)
    {
        return sq;
    }
    else
    {
        return sq.flippedVertically();
    }
}

struct HalfKP {
    static constexpr int NUM_SQ = 64;
    static constexpr int NUM_PT = 10;
    static constexpr int NUM_PLANES = (NUM_SQ * NUM_PT + 1);
    static constexpr int INPUTS = NUM_PLANES * NUM_SQ;

    static constexpr int MAX_ACTIVE_FEATURES = 32;

    static int feature_index(Color color, Square ksq, Square sq, Piece p)
    {
        auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
        return 1 + static_cast<int>(orient(color, sq)) + p_idx * NUM_SQ + static_cast<int>(ksq) * NUM_PLANES;
    }

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        auto& pos = e.pos;
        auto pieces = pos.piecesBB() & ~(pos.piecesBB(Piece(PieceType::King, Color::White)) | pos.piecesBB(Piece(PieceType::King, Color::Black)));
        auto ksq = pos.kingSquare(color);

        // We order the features so that the resulting sparse
        // tensor is coalesced.
        int j = 0;
        for(Square sq : pieces)
        {
            auto p = pos.pieceAt(sq);
            values[j] = 1.0f;
            features[j] = feature_index(color, orient(color, ksq), sq, p);
            ++j;
        }

        return { j, INPUTS };
    }
};

struct HalfKPFactorized {
    // Factorized features
    static constexpr int K_INPUTS = HalfKP::NUM_SQ;  // HalfK的维数
    static constexpr int PIECE_INPUTS = HalfKP::NUM_SQ * HalfKP::NUM_PT;  // HalfP的维数(棋子类型数 * 格子数)
    static constexpr int INPUTS = HalfKP::INPUTS + K_INPUTS + PIECE_INPUTS;  // 分解特征的维数为HalfKP(实特征), HalfK, HalfP的维数总和

    static constexpr int MAX_K_FEATURES = 1;  // 最多动一个王
    static constexpr int MAX_PIECE_FEATURES = 32;  // 最多同时移动32个棋子(从无到有)
    static constexpr int MAX_ACTIVE_FEATURES = HalfKP::MAX_ACTIVE_FEATURES + MAX_K_FEATURES + MAX_PIECE_FEATURES;  // 分解特征的最大活跃特征数为实特征和虚特征的...的总和

    /*
     * 稀疏特征提取的核心函数, 主要作用是将棋盘位置转换为NNUE网络可以处理的稀疏特征表示
    **/
    static std::pair<int, int> fill_features_sparse(
        const TrainingDataEntry& e,  // 训练数据条目, 包含棋盘位置信息
        int* features,               // 输出数组, 存储活跃特征的索引
        float* values,               // 输出数组, 存储活跃特征的值(1.0)
        Color color                  // 当前视角的颜色(白方或黑方)
    )
    {
        // 填写实特征
        auto [start_j, offset] = HalfKP::fill_features_sparse(e, features, values, color);
        
        // 填写虚特征
        int j = start_j;
        auto& pos = e.pos;
        {
            // king square factor
            auto ksq = pos.kingSquare(color);
            features[j] = offset + static_cast<int>(orient(color, ksq));
            values[j] = static_cast<float>(start_j);
            ++j;
        }
        offset += K_INPUTS;
        auto pieces = pos.piecesBB() & ~(pos.piecesBB(Piece(PieceType::King, Color::White)) | pos.piecesBB(Piece(PieceType::King, Color::Black)));

        // We order the features so that the resulting sparse
        // tensor is coalesced. Note that we can just sort
        // the parts where values are all 1.0f and leave the
        // halfk feature where it was.
        for(Square sq : pieces)
        {
            auto p = pos.pieceAt(sq);
            auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
            values[j] = 1.0f;
            features[j] = offset + (p_idx * HalfKP::NUM_SQ) + static_cast<int>(orient(color, sq));
            ++j;
        }

        return { j, INPUTS };  // 第一个元素是激活特征数量, 第二个元素是总特征空间大小
    }
};

struct HalfKA {
    static constexpr int NUM_SQ = 64;  
    static constexpr int NUM_PT = 12;
    static constexpr int NUM_PLANES = (NUM_SQ * NUM_PT + 1);
    static constexpr int INPUTS = NUM_PLANES * NUM_SQ;

    static constexpr int MAX_ACTIVE_FEATURES = 32;

    static int feature_index(Color color, Square ksq, Square sq, Piece p)
    {
        auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
        return 1 + static_cast<int>(orient_flip(color, sq)) + p_idx * NUM_SQ + static_cast<int>(ksq) * NUM_PLANES;
    }

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        auto& pos = e.pos;
        auto pieces = pos.piecesBB();
        auto ksq = pos.kingSquare(color);

        int j = 0;
        for(Square sq : pieces)
        {
            auto p = pos.pieceAt(sq);
            values[j] = 1.0f;
            features[j] = feature_index(color, orient_flip(color, ksq), sq, p);
            ++j;
        }

        return { j, INPUTS };
    }
};

struct HalfKAFactorized {
    // Factorized features
    static constexpr int PIECE_INPUTS = HalfKA::NUM_SQ * HalfKA::NUM_PT;
    static constexpr int INPUTS = HalfKA::INPUTS + PIECE_INPUTS;

    static constexpr int MAX_PIECE_FEATURES = 32;
    static constexpr int MAX_ACTIVE_FEATURES = HalfKA::MAX_ACTIVE_FEATURES + MAX_PIECE_FEATURES;

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        const auto [start_j, offset] = HalfKA::fill_features_sparse(e, features, values, color);
        auto& pos = e.pos;
        auto pieces = pos.piecesBB();

        int j = start_j;
        for(Square sq : pieces)
        {
            auto p = pos.pieceAt(sq);
            auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
            values[j] = 1.0f;
            features[j] = offset + (p_idx * HalfKA::NUM_SQ) + static_cast<int>(orient_flip(color, sq));
            ++j;
        }

        return { j, INPUTS };
    }
};

struct HalfKAv2 {
    static constexpr int NUM_SQ = 64;
    static constexpr int NUM_PT = 11;
    static constexpr int NUM_PLANES = NUM_SQ * NUM_PT;
    static constexpr int INPUTS = NUM_PLANES * NUM_SQ;

    static constexpr int MAX_ACTIVE_FEATURES = 32;

    static int feature_index(Color color, Square ksq, Square sq, Piece p)
    {
        auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
        if (p_idx == 11)
            --p_idx; // pack the opposite king into the same NUM_SQ * NUM_SQ
        return static_cast<int>(orient_flip(color, sq)) + p_idx * NUM_SQ + static_cast<int>(ksq) * NUM_PLANES;
    }

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        auto& pos = e.pos;
        auto pieces = pos.piecesBB();
        auto ksq = pos.kingSquare(color);

        int j = 0;
        for(Square sq : pieces)
        {
            auto p = pos.pieceAt(sq);
            values[j] = 1.0f;
            features[j] = feature_index(color, orient_flip(color, ksq), sq, p);
            ++j;
        }

        return { j, INPUTS };
    }
};

struct HalfKAv2Factorized {
    // Factorized features
    static constexpr int NUM_PT = 12;
    static constexpr int PIECE_INPUTS = HalfKAv2::NUM_SQ * NUM_PT;
    static constexpr int INPUTS = HalfKAv2::INPUTS + PIECE_INPUTS;

    static constexpr int MAX_PIECE_FEATURES = 32;
    static constexpr int MAX_ACTIVE_FEATURES = HalfKAv2::MAX_ACTIVE_FEATURES + MAX_PIECE_FEATURES;

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        const auto [start_j, offset] = HalfKAv2::fill_features_sparse(e, features, values, color);
        auto& pos = e.pos;
        auto pieces = pos.piecesBB();

        int j = start_j;
        for(Square sq : pieces)
        {
            auto p = pos.pieceAt(sq);
            auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
            values[j] = 1.0f;
            features[j] = offset + (p_idx * HalfKAv2::NUM_SQ) + static_cast<int>(orient_flip(color, sq));
            ++j;
        }

        return { j, INPUTS };
    }
};

// ksq must not be oriented
static Square orient_flip_2(Color color, Square sq, Square ksq)
{
    bool h = ksq.file() < fileE;
    if (color == Color::Black)
        sq = sq.flippedVertically();
    if (h)
        sq = sq.flippedHorizontally();
    return sq;
}

struct HalfKAv2_hm {
    static constexpr int NUM_SQ = 64;
    static constexpr int NUM_PT = 11;
    static constexpr int NUM_PLANES = NUM_SQ * NUM_PT;
    static constexpr int INPUTS = NUM_PLANES * NUM_SQ / 2;

    static constexpr int MAX_ACTIVE_FEATURES = 32;

    static constexpr int KingBuckets[64] = {
      -1, -1, -1, -1, 31, 30, 29, 28,
      -1, -1, -1, -1, 27, 26, 25, 24,
      -1, -1, -1, -1, 23, 22, 21, 20,
      -1, -1, -1, -1, 19, 18, 17, 16,
      -1, -1, -1, -1, 15, 14, 13, 12,
      -1, -1, -1, -1, 11, 10, 9, 8,
      -1, -1, -1, -1, 7, 6, 5, 4,
      -1, -1, -1, -1, 3, 2, 1, 0
    };

    static int feature_index(Color color, Square ksq, Square sq, Piece p)
    {
        Square o_ksq = orient_flip_2(color, ksq, ksq);
        auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
        if (p_idx == 11)
            --p_idx; // pack the opposite king into the same NUM_SQ * NUM_SQ
        return static_cast<int>(orient_flip_2(color, sq, ksq)) + p_idx * NUM_SQ + KingBuckets[static_cast<int>(o_ksq)] * NUM_PLANES;
    }

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        auto& pos = e.pos;
        auto pieces = pos.piecesBB();
        auto ksq = pos.kingSquare(color);

        int j = 0;
        for(Square sq : pieces)
        {
            auto p = pos.pieceAt(sq);
            values[j] = 1.0f;
            features[j] = feature_index(color, ksq, sq, p);
            ++j;
        }

        return { j, INPUTS };
    }
};

struct HalfKAv2_hmFactorized {
    // Factorized features
    static constexpr int NUM_PT = 12;
    static constexpr int PIECE_INPUTS = HalfKAv2_hm::NUM_SQ * NUM_PT;
    static constexpr int INPUTS = HalfKAv2_hm::INPUTS + PIECE_INPUTS;

    static constexpr int MAX_PIECE_FEATURES = 32;
    static constexpr int MAX_ACTIVE_FEATURES = HalfKAv2_hm::MAX_ACTIVE_FEATURES + MAX_PIECE_FEATURES;

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        const auto [start_j, offset] = HalfKAv2_hm::fill_features_sparse(e, features, values, color);
        auto& pos = e.pos;
        auto pieces = pos.piecesBB();
        auto ksq = pos.kingSquare(color);

        int j = start_j;
        for(Square sq : pieces)
        {
            auto p = pos.pieceAt(sq);
            auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
            values[j] = 1.0f;
            features[j] = offset + (p_idx * HalfKAv2_hm::NUM_SQ) + static_cast<int>(orient_flip_2(color, sq, ksq));
            ++j;
        }

        return { j, INPUTS };
    }
};

// TODO: 加上自建特征集
struct HalfATA {
    // TODO
    static constexpr int NUM_SQ = 63;
    static constexpr int NUM_PT = 16;
    static constexpr int NUM_ATTACK_BUCKETS = 6;
    static constexpr int NUM_PLANES = NUM_SQ * NUM_PT;
    static constexpr int INPUTS = NUM_PLANES * NUM_ATTACK_BUCKETS;

    static constexpr int MAX_ACTIVE_FEATURES = 32;


    /**
     * 接收局面信息, 根据特征集定义返回单一特征下标
     * @param color 棋子颜色
     * @param attack_bucket 攻击桶
     * @param sq 棋子位置
     * @param p 棋子类型
     * @return 活跃特征下标
     */
    static int feature_index(Color color, int attack_bucket, Square sq, Piece p)
    {
        auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);  // 己方偶数, 对方奇数
        return static_cast<int>(sq) + p_idx * NUM_SQ + attack_bucket * NUM_PLANES;
    }

    /**
     * 从pos对象中解包棋局信息, 分类攻击等级
     */
    static int classify_attack_buckets(const Position &pos) {
        // TODO: 完善stub
        return 0;
    }

    /**
     * 根据训练数据条目信息, 填写特征下标数组和特征值数组, 用于构建SparseBatch
     * @param e 训练数据条目信息
     * @param features 特征下标数组
     * @param values 特征激活值数组
     * @param color 棋子颜色
     * @return {激活值个数, 总特征维数}
     */
    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        auto& pos = e.pos;
        auto pieces = pos.piecesBB();
        int attack_bucket = classify_attack_buckets(pos);

        int j = 0;
        for(Square sq : pieces)
        {
            auto p = pos.pieceAt(sq);
            values[j] = 1.0f;
            features[j] = feature_index(color, attack_bucket, sq, p);
            ++j;
        }

        return { j, INPUTS };
    }
};

struct HalfATAFactorized {
    // TODO
    // Factorized features
    static constexpr int NUM_PT = 16;
    static constexpr int PIECE_INPUTS = HalfATA::NUM_SQ * NUM_PT;
    static constexpr int INPUTS = HalfATA::INPUTS + PIECE_INPUTS;

    static constexpr int MAX_PIECE_FEATURES = 32;
    static constexpr int MAX_ACTIVE_FEATURES = HalfATA::MAX_ACTIVE_FEATURES + MAX_PIECE_FEATURES;

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        const auto [start_j, offset] = HalfATA::fill_features_sparse(e, features, values, color);
        auto& pos = e.pos;
        auto pieces = pos.piecesBB();

        int j = start_j;
        for(Square sq : pieces)
        {
            auto p = pos.pieceAt(sq);
            auto p_idx = static_cast<int>(p.type()) * 2 + (p.color() != color);
            values[j] = 1.0f;
            features[j] = offset + (p_idx * HalfATA::NUM_SQ) + static_cast<int>(sq);
            ++j;
        }

        return { j, INPUTS };
    }
};

/*
 * 特征集抽象类
 * 模板参数为特征集结构体
**/
template <typename T, typename... Ts>
struct FeatureSet
{
    static_assert(sizeof...(Ts) == 0, "Currently only one feature subset supported.");

    static constexpr int INPUTS = T::INPUTS;
    static constexpr int MAX_ACTIVE_FEATURES = T::MAX_ACTIVE_FEATURES;

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        return T::fill_features_sparse(e, features, values, color);
    }
};

/**
 * 一批训练数据的抽象类
 */
struct SparseBatch
{
    static constexpr bool IS_BATCH = true;

    /**
     * 模板参数为特征集结构体
     */
    template <typename... Ts>
    SparseBatch(FeatureSet<Ts...>, const std::vector<TrainingDataEntry>& entries)
    {
        num_inputs = FeatureSet<Ts...>::INPUTS;  // 特征向量维数
        size = entries.size();       // 单个批次内训练数据条目数量
        is_white = new float[size];  // 单批次内的训练数据是否为白色方 (为什么用float不用bool?)
        outcome = new float[size];   // 批次内对局结果
        score = new float[size];     // 批次内引擎分?
        white = new int[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];  // 白色方活跃特征下标
        black = new int[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];
        white_values = new float[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];  // 白色方活跃特征值
        black_values = new float[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];
        psqt_indices = new int[size];
        layer_stack_indices = new int[size];  // 单个batch内每个数据取的bucket

        // 初始化
        num_active_white_features = 0;
        num_active_black_features = 0;
        max_active_features = FeatureSet<Ts...>::MAX_ACTIVE_FEATURES;

        for (std::size_t i = 0; i < size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES; ++i)
            white[i] = -1;
        for (std::size_t i = 0; i < size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES; ++i)
            black[i] = -1;
        for (std::size_t i = 0; i < size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES; ++i)
            white_values[i] = 0.0f;
        for (std::size_t i = 0; i < size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES; ++i)
            black_values[i] = 0.0f;

        for(int i = 0; i < entries.size(); ++i)
        {
            fill_entry(FeatureSet<Ts...>{}, i, entries[i]);
        }
    }

    int num_inputs;
    int size;

    float* is_white;
    float* outcome;
    float* score;
    int num_active_white_features;
    int num_active_black_features;
    int max_active_features;
    int* white;
    int* black;
    float* white_values;
    float* black_values;
    int* psqt_indices;
    int* layer_stack_indices;

    ~SparseBatch()
    {
        delete[] is_white;
        delete[] outcome;
        delete[] score;
        delete[] white;
        delete[] black;
        delete[] white_values;
        delete[] black_values;
        delete[] psqt_indices;
        delete[] layer_stack_indices;
    }

private:

    /**
     * 加载来自单条训练数据的信息
     */
    template <typename... Ts>
    void fill_entry(FeatureSet<Ts...>, int i, const TrainingDataEntry& e)
    {
        // TODO: 涉及pos, piecesBB, 改库
        is_white[i] = static_cast<float>(e.pos.sideToMove() == Color::White);
        outcome[i] = (e.result + 1.0f) / 2.0f;
        score[i] = e.score;
        psqt_indices[i] = (e.pos.piecesBB().count() - 1) / 4;
        layer_stack_indices[i] = psqt_indices[i];
        fill_features(FeatureSet<Ts...>{}, i, e);
    }

    /*
     * 填写活跃特SparseBatch征值, 记录活跃特征索引和数量
    **/
    template <typename... Ts>
    void fill_features(FeatureSet<Ts...>, int i, const TrainingDataEntry& e)
    {
        const int offset = i * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES;
        num_active_white_features +=
            FeatureSet<Ts...>::fill_features_sparse(e, white + offset, white_values + offset, Color::White)
            .first;
        num_active_black_features +=
            FeatureSet<Ts...>::fill_features_sparse(e, black + offset, black_values + offset, Color::Black)
            .first;
    }
};

/*
 * 一个最顶层的抽象基类, 唯一目的是提供一个虚析构函数 virtual ~AnyStream(), 确保能通过指向基类的指针正确析构派生类
**/
struct AnyStream
{
    virtual ~AnyStream() = default;
};

/*
 * 对 AnyStream 的一个具体化，但它本身仍然是一个抽象类, 定义了一个更具体的数据流应该是什么样的
**/
template <typename StorageT>
struct Stream : AnyStream
{
    using StorageType = StorageT;  // 流产生的数据类型

    Stream(int concurrency, const std::vector<std::string>& filenames, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
        m_stream(training_data::open_sfen_input_file_parallel(concurrency, filenames, cyclic, skipPredicate))
    {
    }

    /*
     * 获取数据流的下一项
     * 虚函数, 留给实现类定义具体行为
    **/
    virtual StorageT* next() = 0;

protected:
    // 一个 std::unique_ptr，它指向一个真正负责从文件中读取原始数据的对象。
    // unique_ptr 是一种智能指针，能自动管理内存，当 Stream 对象被销毁时，它指向的 BasicSfenInputStream 对象也会被自动销毁。
    std::unique_ptr<training_data::BasicSfenInputStream> m_stream;
};

/*
 * 一个尝试实现异步数据获取的流
**/
template <typename StorageT>
struct AsyncStream : Stream<StorageT>
{
    using BaseType = Stream<StorageT>;

    AsyncStream(int concurrency, const std::vector<std::string>& filenames, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
        BaseType(1, filenames, cyclic, skipPredicate)
    {
    }

    ~AsyncStream()
    {
        if (m_next.valid())
        {
            delete m_next.get();
        }
    }

protected:
    // std::future: C++并发编程中的一个工具, 代表一个在未来某个时间点才能获得的结果
    // next() 方法可能在返回当前数据块的同时，就在后台启动一个任务去加载下一个数据块，并将这个任务的 std::future 存放在 m_next 中。
    // 这样，当你下次调用 next() 时，数据可能已经准备好了，从而减少了等待时间。
    std::future<StorageT*> m_next;
};

/*
 * 创建一个名为 FeaturedBatchStream 的类。这个类的作用是：
 * 从文件中并行读取原始训练数据
 * 使用多个后台工作线程将原始数据转换成模型接收的特征
 * 将处理好的数据打包成一个个批次 (batch)。
 * 主训练程序可以简单地调用 next() 方法来获取一个准备好的数据批次，而不需要等待文件读取和数据处理，从而大大提高了数据供给效率，避免 GPU/CPU 在训练时空闲。
 * 这整个结构是一个经典的生产者-消费者模式 (Producer-Consumer Pattern)。
 * 生产者：后台的工作线程 (m_workers)，它们负责读取文件、处理数据，并把生产好的数据批次（batch）放入一个队列（m_batches）。
 * 消费者：调用 next() 方法的主线程，它从队列中取走已经准备好的数据批次。
**/
template <typename FeatureSetT, typename StorageT>
struct FeaturedBatchStream : Stream<StorageT>
{
    static_assert(StorageT::IS_BATCH);

    using FeatureSet = FeatureSetT;
    using BaseType = Stream<StorageT>;

    static constexpr int num_feature_threads_per_reading_thread = 2;

    FeaturedBatchStream(int concurrency, const std::vector<std::string>& filenames, int batch_size, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
        BaseType(
            std::max(
                1,
                concurrency / num_feature_threads_per_reading_thread  // 总并发线程数 / 单个读任务需要的线程数
            ),
            filenames,  // 从文件中读取数据, 创建流(此处为SparseBatch组成的流)
            cyclic,
            skipPredicate
        ),
        m_concurrency(concurrency),
        m_batch_size(batch_size)
    {
        m_stop_flag.store(false);

        // 核心: 生产者逻辑
        auto worker = [this]()
        {
            std::vector<TrainingDataEntry> entries;
            entries.reserve(m_batch_size);

            // 没有收到停止信号就一直加载数据
            while(!m_stop_flag.load())
            {
                entries.clear();

                {
                    std::unique_lock lock(m_stream_mutex);  // 给文件流加锁，因为多个线程不能同时读取同一个文件流，需要排队。
                    BaseType::m_stream->fill(entries, m_batch_size);
                    if (entries.empty())
                    {
                        break;
                    }
                }

                auto batch = new StorageT(FeatureSet{}, entries);

                {
                    std::unique_lock lock(m_batch_mutex);
                    m_batches_not_full.wait(lock, [this]() { return m_batches.size() < m_concurrency + 1 || m_stop_flag.load(); });

                    m_batches.emplace_back(batch);

                    lock.unlock();
                    m_batches_any.notify_one();
                }

            }
            m_num_workers.fetch_sub(1);
            m_batches_any.notify_one();
        };

        const int num_feature_threads = std::max(
            1,
            concurrency - std::max(1, concurrency / num_feature_threads_per_reading_thread)
        );

        for (int i = 0; i < num_feature_threads; ++i)
        {
            m_workers.emplace_back(worker);

            // This cannot be done in the thread worker. We need
            // to have a guarantee that this is incremented, but if
            // we did it in the worker there's no guarantee
            // that it executed.
            m_num_workers.fetch_add(1);
        }
    }

    StorageT* next() override
    {
        std::unique_lock lock(m_batch_mutex);
        m_batches_any.wait(lock, [this]() { return !m_batches.empty() || m_num_workers.load() == 0; });

        if (!m_batches.empty())
        {
            auto batch = m_batches.front();
            m_batches.pop_front();

            lock.unlock();
            m_batches_not_full.notify_one();

            return batch;
        }
        return nullptr;
    }

    ~FeaturedBatchStream()
    {
        m_stop_flag.store(true);
        m_batches_not_full.notify_all();

        for (auto& worker : m_workers)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }

        for (auto& batch : m_batches)
        {
            delete batch;
        }
    }

private:
    int m_batch_size;
    int m_concurrency;
    std::deque<StorageT*> m_batches;
    std::mutex m_batch_mutex;
    std::mutex m_stream_mutex;
    std::condition_variable m_batches_not_full;
    std::condition_variable m_batches_any;
    std::atomic_bool m_stop_flag;
    std::atomic_int m_num_workers;

    std::vector<std::thread> m_workers;
};

// Very simple fixed size string wrapper with a stable ABI to pass to python.
/*
 * 一个局面码字符串的封装类
**/
struct Fen
{
    Fen() :
        m_fen(nullptr)
    {
    }

    Fen(const std::string& fen) :
        m_size(fen.size()),
        m_fen(new char[fen.size() + 1])
    {
        std::memcpy(m_fen, fen.c_str(), fen.size() + 1);
    }

    Fen& operator=(const std::string& fen)
    {
        if (m_fen != nullptr)
        {
            delete[] m_fen;
        }

        m_size = fen.size();
        m_fen = new char[fen.size() + 1];
        std::memcpy(m_fen, fen.c_str(), fen.size() + 1);

        return *this;
    }

    ~Fen()
    {
        delete[] m_fen;
    }

private:
    int m_size;
    char* m_fen;
};

/*
 * 一批局面码字符串的封装类
**/
struct FenBatch
{
    FenBatch(const std::vector<TrainingDataEntry>& entries) :
        m_size(entries.size()),
        m_fens(new Fen[entries.size()])
    {
        for (int i = 0; i < m_size; ++i)
        {
            m_fens[i] = entries[i].pos.fen();
        }
    }

    ~FenBatch()
    {
        delete[] m_fens;
    }

private:
    int m_size;
    Fen* m_fens;
};

/*
 * 局面码流(黑魔法)
**/
struct FenBatchStream : Stream<FenBatch>
{
    static constexpr int num_feature_threads_per_reading_thread = 2;

    using BaseType = Stream<FenBatch>;

    FenBatchStream(int concurrency, const std::vector<std::string>& filenames, int batch_size, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
        BaseType(
            std::max(
                1,
                concurrency / num_feature_threads_per_reading_thread
            ),
            filenames,
            cyclic,
            skipPredicate
        ),
        m_concurrency(concurrency),
        m_batch_size(batch_size)
    {
        m_stop_flag.store(false);

        auto worker = [this]()
        {
            std::vector<TrainingDataEntry> entries;
            entries.reserve(m_batch_size);

            while(!m_stop_flag.load())
            {
                entries.clear();

                {
                    std::unique_lock lock(m_stream_mutex);
                    BaseType::m_stream->fill(entries, m_batch_size);
                    if (entries.empty())
                    {
                        break;
                    }
                }

                auto batch = new FenBatch(entries);

                {
                    std::unique_lock lock(m_batch_mutex);
                    m_batches_not_full.wait(lock, [this]() { return m_batches.size() < m_concurrency + 1 || m_stop_flag.load(); });

                    m_batches.emplace_back(batch);

                    lock.unlock();
                    m_batches_any.notify_one();
                }

            }
            m_num_workers.fetch_sub(1);
            m_batches_any.notify_one();
        };

        const int num_feature_threads = std::max(
            1,
            concurrency - std::max(1, concurrency / num_feature_threads_per_reading_thread)
        );

        for (int i = 0; i < num_feature_threads; ++i)
        {
            m_workers.emplace_back(worker);

            // This cannot be done in the thread worker. We need
            // to have a guarantee that this is incremented, but if
            // we did it in the worker there's no guarantee
            // that it executed.
            m_num_workers.fetch_add(1);
        }
    }

    FenBatch* next()
    {
        std::unique_lock lock(m_batch_mutex);
        m_batches_any.wait(lock, [this]() { return !m_batches.empty() || m_num_workers.load() == 0; });

        if (!m_batches.empty())
        {
            auto batch = m_batches.front();
            m_batches.pop_front();

            lock.unlock();
            m_batches_not_full.notify_one();

            return batch;
        }
        return nullptr;
    }

    ~FenBatchStream()
    {
        m_stop_flag.store(true);
        m_batches_not_full.notify_all();

        for (auto& worker : m_workers)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }

        for (auto& batch : m_batches)
        {
            delete batch;
        }
    }

private:
    int m_batch_size;
    int m_concurrency;
    std::deque<FenBatch*> m_batches;
    std::mutex m_batch_mutex;
    std::mutex m_stream_mutex;
    std::condition_variable m_batches_not_full;
    std::condition_variable m_batches_any;
    std::atomic_bool m_stop_flag;
    std::atomic_int m_num_workers;

    std::vector<std::thread> m_workers;
};

struct DataloaderSkipConfig {
    bool filtered;
    int random_fen_skipping;
    bool wld_filtered;
    int early_fen_skipping;
    int simple_eval_skipping;
    int param_index;
};

/*
 * 决定是否跳过某局面的谓词函数
**/
std::function<bool(const TrainingDataEntry&)> make_skip_predicate(DataloaderSkipConfig config)
{
    if (config.filtered || config.random_fen_skipping || config.wld_filtered || config.early_fen_skipping)
    {
        return [
            config,
            prob = double(config.random_fen_skipping) / (config.random_fen_skipping + 1)
            ](const TrainingDataEntry& e){

            // VALUE_NONE from Stockfish.
            // We need to allow a way to skip predetermined positions without
            // having to remove them from the dataset, as otherwise the we lose some
            // compression ability.
            static constexpr int VALUE_NONE = 32002;

            static constexpr double desired_piece_count_weights[33] = {
                1.000000,
                1.121094, 1.234375, 1.339844, 1.437500, 1.527344, 1.609375, 1.683594, 1.750000,
                1.808594, 1.859375, 1.902344, 1.937500, 1.964844, 1.984375, 1.996094, 2.000000,
                1.996094, 1.984375, 1.964844, 1.937500, 1.902344, 1.859375, 1.808594, 1.750000,
                1.683594, 1.609375, 1.527344, 1.437500, 1.339844, 1.234375, 1.121094, 1.000000
            };

            static constexpr double desired_piece_count_weights_total = [](){
                double tot = 0;
                for (auto w : desired_piece_count_weights)
                    tot += w;
                return tot;
            }();

            // keep stats on passing pieces
            static thread_local double alpha = 1;
            static thread_local double piece_count_history_all[33] = {0};
            static thread_local double piece_count_history_passed[33] = {0};
            static thread_local double piece_count_history_all_total = 0;
            static thread_local double piece_count_history_passed_total = 0;

            // max skipping rate
            static constexpr double max_skipping_rate = 10.0;

            auto do_wld_skip = [&]() {
                std::bernoulli_distribution distrib(1.0 - e.score_result_prob());
                auto& prng = rng::get_thread_local_rng();
                return distrib(prng);
            };

            auto do_skip = [&]() {
                std::bernoulli_distribution distrib(prob);
                auto& prng = rng::get_thread_local_rng();
                return distrib(prng);
            };

            auto do_filter = [&]() {
                return (e.isCapturingMove() || e.isInCheck());
            };

            // Allow for predermined filtering without the need to remove positions from the dataset.
            if (e.score == VALUE_NONE)
                return true;

            if (e.ply <= config.early_fen_skipping)
                return true;

            if (config.random_fen_skipping && do_skip())
                return true;

            if (config.filtered && do_filter())
                return true;

            if (config.wld_filtered && do_wld_skip())
                return true;

            if (config.simple_eval_skipping > 0 && std::abs(e.pos.simple_eval()) < config.simple_eval_skipping)
                return true;

            constexpr bool do_debug_print = false;
            if (do_debug_print) {
                if (uint64_t(piece_count_history_all_total) % 10000 == 0) {
                    std::cout << "Total : " << piece_count_history_all_total << '\n';
                    std::cout << "Passed: " << piece_count_history_passed_total << '\n';
                    for (int i = 0; i < 33; ++i)
                        std::cout << i << ' ' << piece_count_history_passed[i] << '\n';
                }
            }

            const int pc = e.pos.piecesBB().count();
            piece_count_history_all[pc] += 1;
            piece_count_history_all_total += 1;

            // update alpha, which scales the filtering probability, to a maximum rate.
            if (uint64_t(piece_count_history_all_total) % 10000 == 0) {
                double pass = piece_count_history_all_total * desired_piece_count_weights_total;
                for (int i = 0; i < 33; ++i)
                {
                    if (desired_piece_count_weights[pc] > 0)
                    {
                        double tmp = piece_count_history_all_total * desired_piece_count_weights[pc] /
                                     (desired_piece_count_weights_total * piece_count_history_all[pc]);
                        if (tmp < pass)
                            pass = tmp;
                    }
                }
                alpha = 1.0 / (pass * max_skipping_rate);
            }

            double tmp = alpha *  piece_count_history_all_total * desired_piece_count_weights[pc] /
                                 (desired_piece_count_weights_total * piece_count_history_all[pc]);
            tmp = std::min(1.0, tmp);
            std::bernoulli_distribution distrib(1.0 - tmp);
            auto& prng = rng::get_thread_local_rng();
            if (distrib(prng))
                return true;

            piece_count_history_passed[pc] += 1;
            piece_count_history_passed_total += 1;

            return false;
        };
    }

    return nullptr;
}

extern "C" {

    /*
     * 使用局面码构造训练批次
    **/
    EXPORT SparseBatch* get_sparse_batch_from_fens(
        const char* feature_set_c,
        int num_fens,
        const char* const* fens,
        int* scores,
        int* plies,
        int* results
    )
    {
        std::vector<TrainingDataEntry> entries;
        entries.reserve(num_fens);
        for (int i = 0; i < num_fens; ++i)
        {
            // TODO: 调用了从fen构造position对象的方法, 需要修改
            auto& e = entries.emplace_back();    // 构造TrainingDataEntry对象
            e.pos = Position::fromFen(fens[i]);  // 用局面码还原出pos对象
            movegen::forEachLegalMove(e.pos, [&](Move m){e.move = m;});
            e.score = scores[i];
            e.ply = plies[i];
            e.result = results[i];
        }

        std::string_view feature_set(feature_set_c);
        if (feature_set == "HalfKP")
        {
            return new SparseBatch(FeatureSet<HalfKP>{}, entries);
        }
        else if (feature_set == "HalfKP^")
        {
            return new SparseBatch(FeatureSet<HalfKPFactorized>{}, entries);
        }
        else if (feature_set == "HalfKA")
        {
            return new SparseBatch(FeatureSet<HalfKA>{}, entries);
        }
        else if (feature_set == "HalfKA^")
        {
            return new SparseBatch(FeatureSet<HalfKAFactorized>{}, entries);
        }
        else if (feature_set == "HalfKAv2")
        {
            return new SparseBatch(FeatureSet<HalfKAv2>{}, entries);
        }
        else if (feature_set == "HalfKAv2^")
        {
            return new SparseBatch(FeatureSet<HalfKAv2Factorized>{}, entries);
        }
        else if (feature_set == "HalfKAv2_hm")
        {
            return new SparseBatch(FeatureSet<HalfKAv2_hm>{}, entries);
        }
        else if (feature_set == "HalfKAv2_hm^")
        {
            return new SparseBatch(FeatureSet<HalfKAv2_hmFactorized>{}, entries);
        }
        // TODO
        else if (feature_set == "HalfATA")
        {
            return new SparseBatch(FeatureSet<HalfATA>{}, entries);
        }
        else if (feature_set == "HalfATA^")
        {
            return new SparseBatch(FeatureSet<HalfATAFactorized>{}, entries);
        }
        fprintf(stderr, "Unknown feature_set %s\n", feature_set_c);
        return nullptr;
    }

    // changing the signature needs matching changes in data_loader/_native.py
    EXPORT FenBatchStream* CDECL create_fen_batch_stream(int concurrency, int num_files, const char* const* filenames, int batch_size, bool cyclic, DataloaderSkipConfig config)
    {
        auto skipPredicate = make_skip_predicate(config);
        auto filenames_vec = std::vector<std::string>(filenames, filenames + num_files);

        return new FenBatchStream(concurrency, filenames_vec, batch_size, cyclic, skipPredicate);
    }

    EXPORT void CDECL destroy_fen_batch_stream(FenBatchStream* stream)
    {
        delete stream;
    }

    // changing the signature needs matching changes in data_loader/_native.py
    EXPORT Stream<SparseBatch>* CDECL create_sparse_batch_stream(
        const char* feature_set_c,
        int concurrency,
        int num_files,
        const char* const* filenames,
        int batch_size,
        bool cyclic,
        DataloaderSkipConfig config
    )
    {
        auto skipPredicate = make_skip_predicate(config);
        auto filenames_vec = std::vector<std::string>(filenames, filenames + num_files);

        std::string_view feature_set(feature_set_c);
        if (feature_set == "HalfKP")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKP>, SparseBatch>(concurrency, filenames_vec, batch_size, cyclic, skipPredicate);
        }
        else if (feature_set == "HalfKP^")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKPFactorized>, SparseBatch>(concurrency, filenames_vec, batch_size, cyclic, skipPredicate);
        }
        else if (feature_set == "HalfKA")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKA>, SparseBatch>(concurrency, filenames_vec, batch_size, cyclic, skipPredicate);
        }
        else if (feature_set == "HalfKA^")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKAFactorized>, SparseBatch>(concurrency, filenames_vec, batch_size, cyclic, skipPredicate);
        }
        else if (feature_set == "HalfKAv2")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKAv2>, SparseBatch>(concurrency, filenames_vec, batch_size, cyclic, skipPredicate);
        }
        else if (feature_set == "HalfKAv2^")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKAv2Factorized>, SparseBatch>(concurrency, filenames_vec, batch_size, cyclic, skipPredicate);
        }
        else if (feature_set == "HalfKAv2_hm")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKAv2_hm>, SparseBatch>(concurrency, filenames_vec, batch_size, cyclic, skipPredicate);
        }
        else if (feature_set == "HalfKAv2_hm^")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKAv2_hmFactorized>, SparseBatch>(concurrency, filenames_vec, batch_size, cyclic, skipPredicate);
        }
        // TODO: 加上自建特征集
        else if (feature_set == "HalfATA")
        {
            return new FeaturedBatchStream<FeatureSet<HalfATA>, SparseBatch>(concurrency, filenames_vec, batch_size, cyclic, skipPredicate);
        }
        else if (feature_set == "HalfATA^")
        {
            return new FeaturedBatchStream<FeatureSet<HalfATAFactorized>, SparseBatch>(concurrency, filenames_vec, batch_size, cyclic, skipPredicate);
        }
        fprintf(stderr, "Unknown feature_set %s\n", feature_set_c);
        return nullptr;
    }

    EXPORT void CDECL destroy_sparse_batch_stream(Stream<SparseBatch>* stream)
    {
        delete stream;
    }

    EXPORT SparseBatch* CDECL fetch_next_sparse_batch(Stream<SparseBatch>* stream)
    {
        return stream->next();
    }

    EXPORT FenBatch* CDECL fetch_next_fen_batch(Stream<FenBatch>* stream)
    {
        return stream->next();
    }

    EXPORT void CDECL destroy_sparse_batch(SparseBatch* e)
    {
        delete e;
    }

    EXPORT void CDECL destroy_fen_batch(FenBatch* e)
    {
        delete e;
    }

}

#if defined(BENCH)

/* benches
   compile and run with:
     g++ -std=c++20 -g3 -O3 -DNDEBUG -DBENCH -march=native training_data_loader.cpp && ./a.out /path/to/binpack
*/

#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>

long long get_rchar_self() {
    std::ifstream io_file("/proc/self/io");
    std::string line;
    while (std::getline(io_file, line)) {
        if (line.rfind("rchar:", 0) == 0) {
            return std::stoll(line.substr(6));
        }
    }
    return -1; // Error or not found
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " file1 [file2 ...]\n";
        return 1;
    }

    const char** files = const_cast<const char**>(&argv[1]);
    int file_count = argc - 1;

#ifdef PGO_BUILD
    const int concurrency = 1;
#else
    const int concurrency = std::thread::hardware_concurrency();
#endif
    // some typical numbers, more skipping means more load
    const int batch_size = 16384;
    const bool cyclic = true;
    const DataloaderSkipConfig config = {
        .filtered = true,
        .random_fen_skipping = 3,
        .wld_filtered = true,
        .early_fen_skipping = 5,
        .simple_eval_skipping = 0,
        .param_index = 0
    };
    // TODO: 修改特征集
    auto stream = create_sparse_batch_stream("HalfKAv2_hm^", concurrency, file_count, files, batch_size, cyclic, config);

    auto t0 = std::chrono::high_resolution_clock::now();

#ifdef PGO_BUILD
    constexpr int iteration_count = 30;
#else
    constexpr int iteration_count = 6000;
#endif

    for (int i = 1; i <= iteration_count; ++i)
    {
        destroy_sparse_batch(stream->next());
        auto t1 = std::chrono::high_resolution_clock::now();
        if (i % 1 == 0)
        {
             double sec = (t1 - t0).count() / 1e9;
             long long bytes = get_rchar_self();
             std::cout << "\rIter:   " << std::fixed << std::setw(8) << i
                       << "       Time(s): " << std::fixed << std::setw(8) << std::setprecision(3) << sec
                       << "       MPos/s:   " << std::fixed << std::setw(8) << std::setprecision(3) << i * batch_size / (sec * 1000 * 1000)
                       << "       It/s:    " << std::fixed << std::setw(8) << std::setprecision(1) << i / sec
                       << "       MB/s:    " << std::fixed << std::setw(8) << std::setprecision(1) << bytes / (sec * 1024 * 1024)
                       << "       B/pos:  " << std::fixed << std::setw(8) << std::setprecision(1) << bytes / (i * batch_size)
                       <<  std::flush;
        }
    }
    std::cout << std::endl;

    return 0;
}

#endif
