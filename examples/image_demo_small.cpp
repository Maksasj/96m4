#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <execution>
#include <sstream>

#include "96m4.h"
#include "utils.hpp"

#include "stb_image.h"

using namespace m964;

auto model_demonstrate(Model& model, const size_t& N_evolution_steps, std::string target_image_filename, const std::string& file_prefix = "") -> void {
    LoadedImage target_image = load_image(target_image_filename);
    auto img_width = target_image.width;
    auto img_height = target_image.height;

    auto buffer = std::vector<std::int32_t> {};
    buffer.resize(img_width * img_height);

    for(size_t x = 0; x < img_width; ++x) {
        for(size_t y = 0; y < img_height; ++y) {
            model.reset_states();

            for (std::int32_t current_step = 0; current_step < N_evolution_steps; ++current_step) {
                model.get_old_state()(0, 0) = static_cast<float>(x) / static_cast<float>(img_width);
                model.get_old_state()(0, 3) = static_cast<float>(y) / static_cast<float>(img_height);
                model.simulate_step_with_biases();
            }

            auto value = model.get_new_state()(3, 3);

            buffer[x + y*img_width] = hsl_to_rgb((1 - value) * 255, 0.5f, 1.0f);
        }
    }

    std::cout << "Exported final demonstration result: " << file_prefix << "result_final.png" << std::endl;
    stbi_write_jpg(("result/" + file_prefix + "result_final.png").c_str(), img_width, img_height, 4, buffer.data(), img_width * sizeof(std::int32_t));
}

auto main() -> std::int32_t {
    try {
        const std::string target_image_filename = "test_image_2.png";

        auto parameters = GeneticAlgorithmTrainingParameters {
            .model_width = 4,
            .model_height = 4,
            .n_evolution_steps = 25,
            .population_size = 100,
            .initial_mutation_strength = 0.1f,
            .target_cost_threshold = 0.5f,
            .max_epochs = 100000,
            .print_interval_epochs = 20
        };

        auto target_image = load_image(target_image_filename);
        if (!target_image.isValid()) {
            std::cerr << "Critical Error: Failed to load or process target image. Exiting." << std::endl;
            return 1;
        }
        std::cout << "Target image '" << target_image_filename << "' loaded: " << target_image.width << "x" << target_image.height << std::endl;

        const auto& target_pixel_data = target_image.data;
        const int img_width = target_image.width;
        const int img_height = target_image.height;

        auto model_cost_function = [&](Model &model) {
            auto accumulated_cost_over_steps = 0.0f;

            for (int i = 0; i < img_width; ++i) {
                for (int j = 0; j < img_height; ++j) {
                    model.reset_states();

                    for (std::int32_t current_step = 0; current_step < parameters.n_evolution_steps; ++current_step) {
                        model.get_old_state()(0, 0) = static_cast<float>(i) / static_cast<float>(img_width);
                        model.get_old_state()(0, 3) = static_cast<float>(j) / static_cast<float>(img_height);
                        model.simulate_step_with_biases();
                        const auto& current_model_state = model.get_new_state(); // Assuming this provides access to cell states
                        const float difference = current_model_state(3, 3) - target_pixel_data[i][j];
                        accumulated_cost_over_steps += difference * difference;
                    }
                }
            }

            return accumulated_cost_over_steps / static_cast<float>(parameters.n_evolution_steps);
        };

        auto best_model = genetic_algorithm_training_hyper(model_cost_function, parameters);

        model_demonstrate(best_model, parameters.n_evolution_steps, target_image_filename, "final_best_");
    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
    }

    std::cout << "\nProgram finished." << std::endl;

    return 0;
}