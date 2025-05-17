#include <iostream>
#include <thread>
#include <chrono>
#include <algorithm>
#include <string>
#include <random>
#include <execution>
#include <mutex>
#include <unordered_set>

#include "96m4.h"
#include "utils.hpp"

#include "games/pong.hpp"

using namespace m964;

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

auto model_cost(Model &model) -> size_t {
    auto game = PongGame();

    auto i = 0;
    auto score = 0;

    while (!game.is_game_over()) {
        game.simulate_frame();

        auto o = i % 2;
        auto n = (i + 1) % 2;

        auto &old_state = model.states[o].fill([&](const auto &x, const auto &y) {
            return game.screen[y][x] ? 1.0f : 0.0f;
        });

        auto &new_state = model.states[n];
        auto &weights = model.weights;

        auto offset = rand_int(-3, 3);
        for (std::int32_t t = 0; t < (24 + offset); ++t) {
            o = i % 2;
            n = (i + 1) % 2;

            old_state = model.states[o];
            new_state = model.states[n];

            calculate_state(new_state, old_state, weights);

            new_state.apply(NormalizeValue());
            new_state.apply(ReluValue<float>{});
            ++i;
        }

        if (new_state(16, 8) < 0.5f) game.paddle_left();
        if (new_state(16, 8) > 0.5f) game.paddle_right();

        ++score;

        if (score > 1000)
            break;
    }

    return score;
}

auto model_demonstrate(Model& model) -> void {
    model.fill_states(0.0f);

    auto game = PongGame();

    std::int32_t i = 0;
    std::int32_t score = 0;

    while (!game.is_game_over()) {
        game.simulate_frame();

        auto o = i % 2;
        auto n = (i + 1) % 2;

        auto &old_state = model.states[o].fill([&](const auto &x, const auto &y) {
            return game.screen[y][x] ? 1.0f : 0.0f;
        });

        auto &new_state = model.states[n];
        auto &weights = model.weights;

        auto offset = rand_int(-3, 3);
        for (std::int32_t t = 0; t < (24 + offset); ++t) {
            o = i % 2;
            n = (i + 1) % 2;

            old_state = model.states[o];
            new_state = model.states[n];

            calculate_state(new_state, old_state, weights);

            new_state.apply(NormalizeValue());
            new_state.apply(ReluValue<float>{});

            ++i;
        }

        ++score;

        if (new_state(16, 8) < 0.5f) game.paddle_left();
        if (new_state(16, 8) > 0.5f) game.paddle_right();

        game.display_buffer();
        export_state_as_image("state.png", new_state);
        export_state_as_image("weights.png", weights);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::cout << score << std::endl;
}

[[noreturn]] auto main(const int argc, char* argv[]) -> std::int32_t {
    auto arguments = std::unordered_set<std::string> {};

    auto executor = ParallelExecutor();

    for (int i = 1; i < argc; ++i)
        arguments.insert(argv[i]);

    auto best_mutex = std::mutex {};
    auto best = Model(32u, 16u);
    auto best_score = 0;

    best.weights.fill([]() {
        return Kernel3<float>().fill(m964::rand_float(-1.0, 1.0f));
    });

    auto generation = 1;
    auto epoch = 0;

    while (true) {
        best.fill_states(0.0f);

        auto models = std::vector<Model>{};
        models.reserve(100);

        for (auto i = 0; i < 100; ++i) {
            auto model = best;

            model.weights.apply(KernelOffset {
                -1.0f / static_cast<float>(generation), 1.0f / static_cast<float>(generation)
            });
            models.push_back(model);
        }

        executor.execute(models.begin(), models.end(), [&](auto &model) {
            auto score = model_cost(model);

            best_mutex.lock();
            if (score > best_score) {
                best_score = score;
                best = model;
                ++generation;
            }
            best_mutex.unlock();
        });

        ++epoch;
        std::cout << "Epoch " << epoch << " (" <<  epoch * 100 << ") with generation " << generation << " with best score " << best_score << "\n";

        if (best_score > 10000)
            break;
    }

    std::cout << "Training is finished !\n";

    while(1)
        model_demonstrate(best);

    return 0;
}
