#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <execution>
#include <sstream>

#include "96m4.h"
#include "utils.hpp"

using namespace m964;

auto model_demonstrate(float x_start, float x_end, float step, int n_evolution_steps, const std::function<float(float)>& studied_function, Model model) {
    std::cout << "Model prediction: [ ";

    auto point = x_start;
    while (point < x_end) {
        model.reset_states();

        for (std::int32_t current_step = 0; current_step < n_evolution_steps; ++current_step) {
            model.get_old_state()(0, 0) = point;
            model.simulate_step_with_biases();
        }

        const auto& current_model_state = model.get_new_state();
        std::cout << "(" << point << ", " << current_model_state(3, 3) << ")\n";

        point += step;
    }
    std::cout << " ] \n";

    std::cout << "Actuall function: [ ";

    point = x_start;
    while (point < x_end) {
        model.reset_states();

        std::cout << "(" << point << ", " << studied_function(point) << ")\n";

        point += step;
    }

    std::cout << " ] \n";
}

auto main() -> std::int32_t {
    try {
        auto parameters = GeneticAlgorithmTrainingParameters {
            .model_width = 4,
            .model_height = 4,
            .n_evolution_steps = 16,
            .population_size = 100,
            .initial_mutation_strength = 0.1f,
            .target_cost_threshold = 0.7f,
            .max_epochs = 100000,
            .print_interval_epochs = 20
        };

        auto x_start = -5.0f;
        auto x_end = 5.0f;
        auto step = 0.1f;

        auto studied_function = [](const float& x) {
            return std::sin(x) + 1;
        };

        auto model_cost_function = [&](Model &model) {
            auto accumulated_cost_over_steps = 0.0f;

            auto point = x_start;
            while (point < x_end) {
                model.reset_states();

                for (std::int32_t current_step = 0; current_step < parameters.n_evolution_steps; ++current_step) {
                    model.get_old_state()(0, 0) = point;
                    model.simulate_step_with_biases();
                }

                const auto& current_model_state = model.get_new_state();
                const float difference = current_model_state(3, 3) - studied_function(point);
                accumulated_cost_over_steps += difference * difference;

                point += step;
            }

            return accumulated_cost_over_steps;
        };

        auto best_model = genetic_algorithm_training_hyper(model_cost_function, parameters);

        model_demonstrate(x_start, x_end, step, parameters.n_evolution_steps, studied_function, best_model);
    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
    }

    std::cout << "\nProgram finished." << std::endl;

    return 0;
}