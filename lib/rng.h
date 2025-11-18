/**
 * 提供一个线程局部随机数生成器（RNG）的工具函数，用于多线程环境下的随机数生成
 * 每个线程有独立的 RNG，避免线程间竞争和同步开销
 */
#pragma once

#include <random>

namespace rng
{
    template <typename RNG = std::mt19937_64>
    auto& get_thread_local_rng(typename RNG::result_type seed = RNG::default_seed)
    {
        static thread_local RNG s_rng(seed);
        return s_rng;
    }
}
