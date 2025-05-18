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

auto model_cost(Model &model, const size_t& steps) -> size_t {
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

        if (model.get_new_state()(16, 8) < 0.5f) game.paddle_left();
        if (model.get_new_state()(16, 8) > 0.5f) game.paddle_right();

        ++score;

        if (score > 1000)
            break;
    }

    return score;
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

        if (model.get_new_state()(16, 8) < 0.5f) game.paddle_left();
        if (model.get_new_state()(16, 8) > 0.5f) game.paddle_right();

        ++score;

        game.display_buffer();
        export_state_as_image("state.png", model.get_new_state());
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
    auto best_score = model_cost(best, 24);

    best.weights.fill([]() {
        return Kernel().fill(m964::rand_float(-1.0, 1.0f));
    });

    auto generation = 1;
    auto epoch = 0;

    while (true) {
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
            auto score = model_cost(model, 24);

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

        if (best_score > 1000)
            break;
    }

    std::cout << "Training is finished !\n";

    while(1)
        model_demonstrate(best, 24);

    return 0;
}
