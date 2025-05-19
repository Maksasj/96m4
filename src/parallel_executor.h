#pragma once

#include <iostream>

#include <algorithm>
#include <random>
#include <execution>
#include <mutex>
#include <thread>

class LinearExecutor {
    public:
        template<typename RandomAccessIterator, typename Func>
        void execute(RandomAccessIterator first, RandomAccessIterator last, Func func) const {
            for (auto it = first; it != last; ++it) {
                func(*it);
            }
        }
};

class ParallelExecutor {
    public:
        explicit ParallelExecutor(const std::size_t num_threads = std::thread::hardware_concurrency()) : num_threads(num_threads > 0 ? num_threads : 1) {
            if (num_threads == 0)
                std::cerr << "Warning: hardware_concurrency returned 0. Using 1 thread." << std::endl;
        }

        template<typename RandomAccessIterator, typename Func>
        void execute(RandomAccessIterator first, RandomAccessIterator last, Func func) const {
            const auto total_elements = std::distance(first, last);

            if (total_elements == 0)
                return;

            const auto actual_num_threads = std::min((unsigned int)total_elements, num_threads);
            const auto chunk_size = total_elements / actual_num_threads;
            const auto remainder = total_elements % actual_num_threads;

            std::vector<std::thread> threads;
            threads.reserve(actual_num_threads);

            auto current_first = first;

            for (unsigned int i = 0; i < actual_num_threads; ++i) {
                auto current_last = current_first;
                std::advance(current_last, chunk_size + (i < remainder ? 1 : 0));

                threads.emplace_back([current_first, current_last, func]() {
                    for (auto it = current_first; it != current_last; ++it) {
                        func(*it);
                    }
                });

                current_first = current_last;
            }

            for (auto& thread : threads)
                if (thread.joinable())
                    thread.join();
        }

    private:
        unsigned int num_threads;
};