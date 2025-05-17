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

#include "games/dodge.hpp"

using namespace m964;

auto model_cost(Model &model) -> size_t {
    auto game = DodgeGame();

    std::size_t i = 0;
    std::size_t score = 0;

    while (!game.is_game_over()) {
        game.simulate_frame();

        auto o = i % 2;
        auto n = (i + 1) % 2;

        auto &old_state = model.states[o].fill([&](const auto &x, const auto &y) {
            return game.screen[y][x] ? 1.0f : 0.0f;
        });

        auto &new_state = model.states[n];
        auto &weights = model.weights;

        for (std::int32_t t = 0; t < 24; ++t) {
            o = i % 2;
            n = (i + 1) % 2;

            old_state = model.states[o];
            new_state = model.states[n];
            weights = model.weights;

            calculate_state(new_state, old_state, weights);

            new_state.apply(NormalizeValue());
            new_state.apply(ReluValue<float>{});
            ++i;
        }

        // if (new_state(16, 8) < 0.5f) game.paddle_left();
        // if (new_state(16, 8) > 0.5f) game.paddle_right();

        ++score;

        if (score > 1000)
            break;
    }

    return score;
}

auto model_demonstrate(Model& model) -> void {
    auto &weights = model.weights;
    export_state_as_image("weights.png", weights);

    model.fill_states(0.0f);

    auto game = DodgeGame();

    std::int32_t i = 0;
    while (!game.is_game_over()) {
        game.simulate_frame();

        auto o = i % 2;
        auto n = (i + 1) % 2;

        auto &old_state = model.states[o].fill([&](const auto &x, const auto &y) {
            return game.screen[y][x] ? 1.0f : 0.0f;
        });

        auto &new_state = model.states[n];

        for (std::int32_t t = 0; t < 24; ++t) {
            o = i % 2;
            n = (i + 1) % 2;

            old_state = model.states[o];
            new_state = model.states[n];
            weights = model.weights;

            calculate_state(new_state, old_state, weights);

            new_state.apply(NormalizeValue());
            new_state.apply(ReluValue<float>{});

            export_state_as_image("state.png", new_state);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            ++i;
        }

        // if (new_state(16, 8) < 0.5f) game.paddle_left();
        // if (new_state(16, 8) > 0.5f) game.paddle_right();

        game.display_buffer();
    }
}

[[noreturn]] auto main(const int argc, char* argv[]) -> std::int32_t {
    auto arguments = std::unordered_set<std::string> {};

    for (int i = 1; i < argc; ++i)
        arguments.insert(argv[i]);

    auto best_mutex = std::mutex {};
    auto best = Model(12u, 32u);
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

        std::for_each(std::execution::par, models.begin(), models.end(), [&](auto &model) {
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
        std::cout << "Epoch " << epoch << ", " << "with best score " << best_score << "\n";

        if (best_score > 1000)
            break;
    }

    std::cout << "Training is finished !\n";
    model_demonstrate(best);

    return 0;
}
