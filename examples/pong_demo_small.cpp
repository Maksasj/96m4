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

auto model_test(Model &model, const size_t& steps) -> std::size_t {
    auto game = new PongGame();
    auto score = 0;

    while (!game->is_game_over()) {
        game->simulate_frame();

        model.reset_states();
        model.get_old_state()(0, 1) = static_cast<float>(game->get_ball_position().first) / 32.0f;
        model.get_old_state()(1, 1) = static_cast<float>(game->get_ball_position().second) / 16.0f;
        model.get_old_state()(2, 1) = static_cast<float>(game->get_paddle_position().first) / 32.0f;

        // const auto offset = rand_int(-1, 1);
        for (std::int32_t t = 0; t < steps; ++t)
            model.simulate_step();

        const auto sample = model.get_new_state()(1, 0);

        if (sample < 0.5f) game->paddle_left();
        if (sample > 0.5f) game->paddle_right();

        ++score;
    }

    delete game;
    return score;
}

auto model_cost(Model &model, const size_t& steps) -> float {
    auto game = new PongGame();
    auto score = 0;
    auto cost = 0.0f;

    while (!game->is_game_over()) {
        game->simulate_frame();

        model.reset_states();
        model.get_old_state()(0, 1) = static_cast<float>(game->get_ball_position().first) / 32.0f;
        model.get_old_state()(1, 1) = static_cast<float>(game->get_ball_position().second) / 16.0f;
        model.get_old_state()(2, 1) = static_cast<float>(game->get_paddle_position().first) / 32.0f;

        // const auto offset = rand_int(-1, 1);
        for (std::int32_t t = 0; t < steps; ++t)
            model.simulate_step();

        const auto sample = model.get_new_state()(1, 0);
        const auto expected = game->paddle_prediction();
        cost += std::fabs(sample - expected);

        if (expected < 0.5f) game->paddle_left();
        if (expected > 0.5f) game->paddle_right();

        ++score;

        if (score > 1000)
            break;
    }

    delete game;
    return cost;
}

auto model_demonstrate(Model& model, const size_t& steps) -> void {
    auto game = new PongGame();
    auto score = 0;

    while (!game->is_game_over()) {
        game->simulate_frame();

        model.reset_states();
        model.get_old_state()(0, 1) = static_cast<float>(game->get_ball_position().first) / 32.0f;
        model.get_old_state()(1, 1) = static_cast<float>(game->get_ball_position().second) / 16.0f;
        model.get_old_state()(2, 1) = static_cast<float>(game->get_paddle_position().first) / 32.0f;

        // const auto offset = rand_int(-1, 1);
        for (std::int32_t t = 0; t < steps; ++t)
            model.simulate_step();

        const auto sample = model.get_new_state()(1, 0);
        if (sample < 0.5f) game->paddle_left();
        if (sample > 0.5f) game->paddle_right();

        ++score;

        game->display_buffer();
        export_state_as_image("state.png", model.get_new_state());
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    delete game;

    std::cout << score << std::endl;
}

auto main() -> std::int32_t {
    auto arguments = std::unordered_set<std::string> {};

    auto best_mutex = std::mutex {};
    auto best = Model(3, 2);
    auto found_best = false;
    auto best_cost = model_cost(best, 5);

    best.weights.fill([]() {
        return Kernel().fill(m964::rand_float(-1.0, 1.0f));
    });

    auto generation = 1;
    auto epoch = 0;

    while (true) {
        auto models = std::vector<Model>{};
        models.reserve(1000);

        for (auto i = 0; i < 1000; ++i) {
            auto model = best;

            model.weights.apply(KernelOffset {
                -1.0f * (1.0f / static_cast<float>(generation)), 1.0f * (1.0f / static_cast<float>(generation))
            });

            models.push_back(model);
        }

        auto executor = ParallelExecutor();
        executor.execute(models.begin(), models.end(), [&](auto &model) {
            auto cost = model_cost(model, 5);

            best_mutex.lock();
            if (cost < best_cost) {
                best_cost = cost;
                best = model;
                found_best = true;
            }
            best_mutex.unlock();
        });

        if (found_best) {
            ++generation;
            found_best = false;
        }

        ++epoch;
        std::cout << "Epoch " << epoch << " (" <<  epoch * 1000 << ") with generation " << generation << " with best cost " << best_cost <<  "\n";

        if (best_cost < 100.0f)
            break;
    }

    std::cout << "Training is finished !\n";

    while(1)
        model_demonstrate(best, 5);

    return 0;
}
