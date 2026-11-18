#pragma once

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

/*
 * Minimal deterministic fork-join helper.
 *
 * The kernels in this library are pure functions of explicit inputs, so the
 * cheapest way to use several cores without adding OpenMP/TBB as a dependency
 * is a static block partition.  Two properties matter for reproducibility:
 *
 * 1. Every index is processed exactly once by an independent call of fn.
 * 2. The partition depends only on (count, threads), never on scheduling, so
 *    a caller that accumulates per-index results in index order observes the
 *    same values on every run.
 *
 * Set RFLUX_THREADS=N to cap the worker count; RFLUX_THREADS=1 forces the
 * serial path.  The default cap is min(hardware_concurrency, 16).
 */
namespace rflux::detail {

inline unsigned configured_thread_limit() {
    const char* raw = std::getenv("RFLUX_THREADS");
    if (raw == nullptr || *raw == '\0') return 0;
    char* end = nullptr;
    const long value = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0' || value < 1) return 0;
    return static_cast<unsigned>(std::min<long>(value, 64));
}

inline unsigned default_thread_limit() {
    static const unsigned limit = [] {
        const unsigned from_env = configured_thread_limit();
        if (from_env != 0) return from_env;
        const unsigned hardware = std::thread::hardware_concurrency();
        return hardware == 0 ? 1u : std::min(hardware, 16u);
    }();
    return limit;
}

inline std::size_t worker_count(std::size_t count,
                                std::size_t max_threads = 0) {
    const std::size_t limit =
        max_threads == 0 ? default_thread_limit()
                         : std::min<std::size_t>(max_threads,
                                                 default_thread_limit());
    return std::max<std::size_t>(1, std::min(count, limit));
}

/* Nested calls (a parallel kernel invoking another one) stay serial to avoid
 * oversubscribing the machine. */
inline thread_local unsigned parallel_depth = 0;

template <typename Function>
void parallel_for(std::size_t count, Function&& function,
                  std::size_t max_threads = 0) {
    const std::size_t threads =
        parallel_depth == 0 ? worker_count(count, max_threads) : 1;
    if (threads <= 1) {
        ++parallel_depth;
        try {
            for (std::size_t index = 0; index < count; ++index) function(index);
        } catch (...) {
            --parallel_depth;
            throw;
        }
        --parallel_depth;
        return;
    }

    const std::size_t chunk = count / threads + (count % threads != 0);
    std::exception_ptr failure;
    std::mutex failure_mutex;
    auto run = [&](std::size_t begin, std::size_t end) {
        ++parallel_depth;
        try {
            for (std::size_t index = begin; index < end; ++index) {
                function(index);
            }
        } catch (...) {
            std::lock_guard<std::mutex> guard(failure_mutex);
            if (!failure) failure = std::current_exception();
        }
        --parallel_depth;
    };

    std::vector<std::thread> workers;
    workers.reserve(threads - 1);
    for (std::size_t thread = 1; thread < threads; ++thread) {
        const std::size_t begin = thread * chunk;
        const std::size_t end = std::min(count, begin + chunk);
        if (begin >= end) break;
        workers.emplace_back(run, begin, end);
    }
    run(0, std::min(count, chunk));
    for (std::thread& worker : workers) worker.join();
    if (failure) std::rethrow_exception(failure);
}

}  // namespace rflux::detail
