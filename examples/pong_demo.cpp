#include <iostream>
#include <thread>
#include <chrono>
#include <string>
#include <random>
#include <execution>
#include <mutex>

#include "96m4.h"
#include "utils.hpp"

#include "games/pong.hpp"

using namespace m964;

auto model_demonstrate(Model& model, const size_t& steps) -> void {
    auto game = PongGame();
    auto score = 0;

    while (!game.is_game_over()) {
        game.simulate_frame();

        model.get_old_state().fill([&](const auto &x, const auto &y) {
            return game.screen[y][x] ? 1.0f : 0.0f;
        });

        for (std::int32_t t = 0; t < steps; ++t)
            model.simulate_step_with_biases();

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
    try {
        auto parameters = GeneticAlgorithmTrainingParameters {
            .model_width = 32,
            .model_height = 16,
            .n_evolution_steps = 24,
            .population_size = 10,
            .initial_mutation_strength = 1.0f,
            .target_cost_threshold = 10.0f,
            .max_epochs = 100000,
            .print_interval_epochs = 20
        };

        auto model_cost_function = [&](Model &model) {
            auto game = PongGame();
            auto score = 0;
            auto cost = 0.0f;

            while (!game.is_game_over()) {
                game.simulate_frame();

                model.get_old_state().fill([&](const auto &x, const auto &y) {
                    return game.screen[y][x] ? 1.0f : 0.0f;
                });

                for (std::int32_t t = 0; t < parameters.n_evolution_steps; ++t)
                    model.simulate_step_with_biases();

                const auto sample = model.get_new_state()(16, 8);
                const auto expected = game.paddle_prediction();
                cost += std::fabs(expected - sample);

                if (expected < 0.5f) game.paddle_left();
                if (expected > 0.5f) game.paddle_right();

                ++score;

                if (score > 500)
                    break;
            }

            return cost;
        };

        auto best_model = genetic_algorithm_training_hyper(model_cost_function, parameters);

        while (1)
            model_demonstrate(best_model, parameters.n_evolution_steps);
    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
    }

    std::cout << "\nProgram finished." << std::endl;

    return 0;
}
