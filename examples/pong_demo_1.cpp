#include <iostream>
#include <thread>
#include <chrono>
#include <string>
#include <random>
#include <execution>
#include <mutex>
#include <unordered_set>

#include "96m4.h"
#include "utils.hpp"

#include "games/pong.hpp"

using namespace m964;

auto model_cost(Model &model, const size_t& steps) -> float {
    auto game = PongGame();
    auto score = 0;
    auto cost = 0.0f;

    while (!game.is_game_over()) {
        game.simulate_frame();

        model.reset_states();
        model.get_old_state().fill([&](const auto &x, const auto &y) {
            return game.screen[y][x] ? 1.0f : 0.0f;
        });

        auto offset = rand_int(-3, 3);
        for (std::int32_t t = 0; t < (steps + offset); ++t)
            model.simulate_step();

        const auto sample = model.get_new_state()(16, 8);
        const auto expected = game.paddle_prediction();
        cost += 1 - std::fabs(expected - sample);

        if (sample < 0.5f) game.paddle_left();
        if (sample > 0.5f) game.paddle_right();

        ++score;

        if (score > 1000)
            break;
    }

    return cost;
}

auto model_demonstrate(Model& model, const size_t& steps) -> void {
    auto game = PongGame();
    auto score = 0;

    while (!game.is_game_over()) {
        game.simulate_frame();

        model.reset_states();
        model.get_old_state().fill([&](const auto &x, const auto &y) {
            return game.screen[y][x] ? 1.0f : 0.0f;
        });

        auto offset = rand_int(-3, 3);
        for (std::int32_t t = 0; t < (steps + offset); ++t)
            model.simulate_step();

        const auto sample = model.get_new_state()(16, 8);
        if (sample < 0.5f) game.paddle_left();
        if (sample > 0.5f) game.paddle_right();

        ++score;

        game.display_buffer();
        export_state_as_image("state.png", model.get_new_state());
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::cout << score << std::endl;
}

[[noreturn]] auto main() -> std::int32_t {
    auto steps = 24;

    auto best_mutex = std::mutex {};
    auto best = Model(32u, 16u);
    auto found_best = false;
    auto best_cost = model_cost(best, steps);

    best.weights.fill([]() {
        return Kernel().fill(m964::rand_float(-1.0, 1.0f));
    });

    auto generation = 1;
    auto epoch = 0;

    while (true) {
        const auto mutation_rate = 1.0f / static_cast<float>(generation);

        auto models = std::vector<Model>{};
        models.reserve(100);

        for (auto i = 0; i < 100; ++i) {
            auto model = best;

            model.weights.apply(KernelOffset {
                -1.0f * mutation_rate, 1.0f * mutation_rate
            });

            models.push_back(model);
        }

        auto executor = ParallelExecutor();
        executor.execute(models.begin(), models.end(), [&](auto &model) {
            auto cost = model_cost(model, steps);

            best_mutex.lock();
            if (cost > best_cost) {
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
        std::cout << "Epoch " << epoch << " (" <<  epoch * 100 << ") with generation " << generation << " with best cost " << best_cost <<  "\n";

        if (best_cost > 500)
            break;
    }

    std::cout << "Training is finished !\n";

    while(1)
        model_demonstrate(best, 24);

    return 0;
}
