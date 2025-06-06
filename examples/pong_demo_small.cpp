#include <iostream>

#include <chrono>
#include <algorithm>
#include <string>
#include <random>
#include <execution>
#include <mutex>

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
        model.get_old_state()(0, 1) = static_cast<float>(game.get_ball_position().first) / 32.0f;
        model.get_old_state()(1, 1) = static_cast<float>(game.get_ball_position().second) / 16.0f;
        model.get_old_state()(2, 1) = static_cast<float>(game.get_paddle_position().first) / 32.0f;

        const auto offset = rand_int(-1, 1);
        for (std::int32_t t = 0; t < (steps + offset); ++t)
            model.simulate_step();

        const auto sample = model.get_new_state()(1, 0);
        const auto expected = game.paddle_prediction();
        const auto diff = expected - sample;
        cost += diff * diff;

        if (expected < 0.5f) game.paddle_left();
        if (expected > 0.5f) game.paddle_right();

        ++score;

        if (score > 1000)
            break;
    }

    return cost / static_cast<float>(steps);
}

auto model_demonstrate(Model& model, const size_t& steps) -> void {
    auto game = PongGame();
    auto score = 0;

    while (!game.is_game_over()) {
        game.simulate_frame();

        model.reset_states();
        model.get_old_state()(0, 1) = static_cast<float>(game.get_ball_position().first) / 32.0f;
        model.get_old_state()(1, 1) = static_cast<float>(game.get_ball_position().second) / 16.0f;
        model.get_old_state()(2, 1) = static_cast<float>(game.get_paddle_position().first) / 32.0f;

        const auto offset = rand_int(-1, 1);
        for (std::int32_t t = 0; t < (steps + offset); ++t)
            model.simulate_step();

        const auto sample = model.get_new_state()(1, 0);
        if (sample < 0.5f) game.paddle_left();
        if (sample > 0.5f) game.paddle_right();

        ++score;

        game.display_buffer();
        export_state_as_image("state.png", model.get_new_state());
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::cout << score << std::endl;
}

auto main() -> std::int32_t {
    auto steps = 5;

    auto best_mutex = std::mutex {};
    auto best = Model(3, 2);
    auto found_best = false;

    best.weights.fill([]() {
        return Kernel().fill(rand_float(-1.0, 1.0f));
    });

    auto best_cost = model_cost(best, steps);

    auto generation = 1;
    auto epoch = 0;

    while (true) {
        const auto mutation_rate = 1.0f / static_cast<float>(generation);

        auto models = std::vector<Model>{};
        models.reserve(1000);

        for (auto i = 0; i < 1000; ++i) {
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

        if (best_cost < 25.0f)
            break;
    }

    std::cout << "Training is finished !\n";

    while(true)
        model_demonstrate(best, steps);

    return 0;
}
