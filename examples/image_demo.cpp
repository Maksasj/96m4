#include <iostream>
#include <chrono>
#include <string>
#include <execution>
#include <vector>

#include "96m4.h"
#include "utils.hpp"
#include "stb_image.h"

using namespace m964;

auto model_demonstrate(Model& model, const size_t& N_evolution_steps, int img_width, int img_height, const std::string& file_prefix = "") -> void {
    model.reset_states();
    model.get_old_state().fill([]() {
        return 1.0f;
        // return rand_float(0.0f, 1.0f); // Again, consider specific seed states if desired
    });

    std::cout << "Demonstrating model (" << img_width << "x" << img_height << ") for "
              << N_evolution_steps << " steps. File prefix: " << file_prefix << std::endl;

    for (std::int32_t t = 0; t < N_evolution_steps; ++t) {
        model.simulate_step_with_biases();
        // Assuming export_state_as_image can handle model.get_new_state()
        // It would be ideal if Model knew its own dimensions or if export_state_as_image took them.
        export_state_as_image("result/" + file_prefix + "state_step_" + std::to_string(t) + ".png", model.get_new_state());
        if (t % 10 == 0 || t == N_evolution_steps -1 ) { // Print progress less often
            std::cout << "  Exported " << file_prefix << "state_step_" << std::to_string(t) << ".png" << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Adjusted for potentially better viewing
    }

    export_state_as_image("result/" + file_prefix + "result_final.png", model.get_new_state());
    std::cout << "Exported final demonstration result: " << file_prefix << "result_final.png" << std::endl;
}

auto main() -> std::int32_t {
    try {
        const std::string target_image_filename = "test_image_2.png";

        auto target_image = load_image(target_image_filename);
        if (!target_image.isValid()) {
            std::cerr << "Critical Error: Failed to load or process target image. Exiting." << std::endl;
            return 1;
        }
        std::cout << "Target image '" << target_image_filename << "' loaded: " << target_image.width << "x" << target_image.height << std::endl;

        const auto& target_pixel_data = target_image.data;
        const int img_width = target_image.width;
        const int img_height = target_image.height;

        auto parameters = GeneticAlgorithmTrainingParameters {
            .model_width = static_cast<size_t>(img_width),
            .model_height = static_cast<size_t>(img_height),
            .n_evolution_steps = 25,
            .population_size = 100,
            .initial_mutation_strength = 0.1f,
            .target_cost_threshold = 1.0f,
            .max_epochs = 10000,
            .print_interval_epochs = 20
        };

        auto model_cost_function = [&](Model &model) {
            model.reset_states();

            model.get_old_state().fill([]() {
                return 1.0f;
            });

            auto accumulated_cost_over_steps = 0.0f;

            for (std::int32_t current_step = 0; current_step < parameters.n_evolution_steps; ++current_step) {
                model.simulate_step_with_biases();
                const auto& current_model_state = model.get_new_state();
                float cost_for_this_step = 0.0f;

                for (int i = 0; i < img_width; ++i) {
                    for (int j = 0; j < img_height; ++j) {
                        const float target_value = target_pixel_data[i][j];
                        const float model_output_value = current_model_state(i, j);
                        const float difference = model_output_value - target_value;
                        cost_for_this_step += difference * difference;
                    }
                }
                accumulated_cost_over_steps += cost_for_this_step;
            }

            if (parameters.n_evolution_steps == 0) return 0.0f;

            return accumulated_cost_over_steps / static_cast<float>(parameters.n_evolution_steps);
        };

        auto best_model = genetic_algorithm_training_hyper(model_cost_function, parameters);

        model_demonstrate(best_model, parameters.n_evolution_steps, img_width, img_height, "final_best_");
    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
    }

    std::cout << "\nProgram finished." << std::endl;

    return 0;
}

/*
// Assuming Model class has methods like width() and height(), or these are passed
// Ensure model dimensions match target_image_obj dimensions before calling.
auto model_cost(Model &model, const size_t& N_evolution_steps, // Number of steps for the NCA to evolve
                const LoadedImage& target_image_obj) -> float {

}



// ... (includes, using namespace m964, struct LoadedImage, load_image, model_cost, model_demonstrate)

auto main() -> std::int32_t {
    std::cout << "--- Neural Cellular Automata Training Program ---" << std::endl;

    // --- Key Training Hyperparameters ---
    const std::string target_image_filename = "test_image_2.png";
    size_t N_evolution_steps = 10;       // Number of steps for NCA to evolve. 10 is often too few. Try 48-96.
    const int population_size = 100;     // Number of candidate models per generation.
    const float initial_mutation_strength = 0.1f; // Initial range for mutations (e.g., +/- this value). Tune this.
    const float target_cost_threshold = 10.0f; // Adjust based on cost scale and desired accuracy.
    const int max_epochs = 10000;        // Max training epochs to prevent infinite loops.
    const int print_interval_epochs = 20; // How often to print status if no improvement.

    // TODO: Implement proper random number generation and seeding for reproducibility.
    // e.g., using <random>
    // std::mt19937 rng(std::random_device{}());
    // auto uniform_dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);
    // And then use this for rand_float and weight initialization.

    // --- 1. Load Target Image ---
    LoadedImage target_image = load_image(target_image_filename);
    if (!target_image.isValid()) {
        std::cerr << "Critical Error: Failed to load or process target image. Exiting." << std::endl;
        return 1;
    }
    std::cout << "Target image '" << target_image_filename << "' loaded: "
              << target_image.width << "x" << target_image.height << std::endl;

    // --- 2. Initialize Model ---
    // IMPORTANT: This assumes your Model class (from "96m4.h") can be constructed
    // with dynamic dimensions (width, height). If it's fixed-size, you must
    // resize/crop the target_image to match the Model's expected dimensions.
    auto best_model = Model(target_image.width, target_image.height);
    std::cout << "Initialized base model with dimensions: "
              << target_image.width << "x" << target_image.height << std::endl;

    // Initialize model weights.
    // TODO: This is a critical step. Review "Model Architecture" and "Initialization"
    // from the previous detailed guidance for how to best initialize weights within your Model's MLP.
    // The `Kernel().fill(...)` structure is specific to your `96m4.h`.
    // A smaller initial range for weights is often better, e.g., Kaiming/Xavier initialization.

    best_model.bias_layer.fill([&]() { // Assuming rand_float can be used here effectively
        return rand_float(-initial_mutation_strength, initial_mutation_strength); // Example: smaller initial weight range
    });

    best_model.weights.fill([&]() { // Assuming rand_float can be used here effectively
        return Kernel().fill(rand_float(-initial_mutation_strength, initial_mutation_strength)); // Example: smaller initial weight range
    });

    auto best_cost = model_cost(best_model, N_evolution_steps, target_image);
    auto prev_cost = best_cost;

    std::cout << "Initial model cost: " << best_cost << std::endl;

    // --- 3. Evolutionary Training Loop ---
    auto best_mutex = std::mutex{};
    bool found_new_best_this_epoch = false;
    long long generation_count = 1; // Increments when a better model is found
    long long epoch_count = 0;

    // TODO: Consider Curriculum Learning:
    // 1. Start with a smaller `N_evolution_steps` and gradually increase it.
    // 2. Start with simpler or downscaled versions of the `target_image`.

    // TODO: For robustness, explore "Pool Training" / "Experience Replay" where you
    // sometimes damage states and train the NCA to regenerate. (More complex to add here)

    std::cout << "\n--- Starting Training ---" << std::endl;
    std::cout << "N_evolution_steps: " << N_evolution_steps
              << ", Population: " << population_size
              << ", Target Cost: < " << target_cost_threshold
              << ", Max Epochs: " << max_epochs << std::endl;

    while (epoch_count < max_epochs) {
        // Mutation rate can adapt. A common strategy is to decrease it.
        // `1.0f / generation_count` makes it decrease quickly.
        // Using a fixed or slowly decaying `current_mutation_strength` might be more stable.
        const float current_mutation_strength = initial_mutation_strength / std::sqrt(static_cast<float>(generation_count));
        // const float current_mutation_strength = initial_mutation_strength * std::pow(0.999f, epoch_count); // Alternative decay

        std::vector<Model> current_population; // Create vector directly

        for (int i = 0; i < population_size; ++i) {
            current_population.push_back(best_model); // Assumes Model has a proper copy constructor

            // TODO: The mutation mechanism via `KernelOffset` and `apply` is specific to `96m4.h`.
            // Ensure this mutation is effective. It might need to be more targeted or
            // apply different strategies (e.g., perturbing a subset of weights).

            current_population[i].bias_layer.apply([&](auto& value) {
                value += rand_float(-1.0f, 1.0f) * current_mutation_strength;
            });

            current_population[i].weights.apply(KernelOffset{
                -current_mutation_strength, current_mutation_strength
            });
        }

        found_new_best_this_epoch = false;
        ParallelExecutor executor; // Assuming this is your parallel execution utility

        auto epoch_start_time = std::chrono::high_resolution_clock::now();

        executor.execute(current_population.begin(), current_population.end(),
            [&](Model &candidate_model) {
            // `model_cost` re-initializes the model's grid state with random values.
            // If a specific seed/starting pattern (e.g. single central pixel) is desired
            // for each evaluation, ensure `model_cost` (or a variant) handles that.
            float candidate_cost = model_cost(candidate_model, N_evolution_steps, target_image);

            std::lock_guard<std::mutex> lock(best_mutex);
            if (candidate_cost < best_cost) {
                prev_cost = best_cost;
                best_cost = candidate_cost;
                best_model = candidate_model; // Assumes Model assignment is efficient
                found_new_best_this_epoch = true;
            }
        });

        auto epoch_end_time = std::chrono::high_resolution_clock::now();
        auto epoch_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end_time - epoch_start_time).count();

        if (found_new_best_this_epoch) {
            ++generation_count;

            double costChangePerEpoch = prev_cost - best_cost;
            double totalCostReductionNeeded = best_cost - target_cost_threshold;
            auto predicted_epochs = static_cast<std::size_t>(std::ceil(totalCostReductionNeeded / costChangePerEpoch));

            printf("Epoch %lld | Gen %lld | New Best Cost: %.6f | Predicted : %lld | Mut.Strength: %.4f | Time: %lldms *Improvement!*\n",
                   epoch_count, generation_count, best_cost, predicted_epochs, current_mutation_strength, epoch_duration_ms);

            // Optional: Save the new best model periodically
            // best_model.save("best_model_gen_" + std::to_string(generation_count) + ".dat"); // Needs Model::save
            // Optional: Demonstrate intermediate best models
            if (generation_count % 10 == 0) { // Demonstrate every 10 generations
                 // model_demonstrate(best_model, N_evolution_steps, target_image.width, target_image.height, "gen_" + std::to_string(generation_count) + "_");
            }
        } else {
            if (epoch_count % print_interval_epochs == 0) {
                 printf("Epoch %lld | Gen %lld | Best Cost: %.6f | Mut.Strength: %.4f | Time: %lldms\n",
                       epoch_count, generation_count, best_cost, current_mutation_strength, epoch_duration_ms);
            }
        }

        ++epoch_count;

        if (best_cost < target_cost_threshold) {
            std::cout << "\nTarget cost threshold (" << target_cost_threshold << ") reached at epoch " << epoch_count << "!" << std::endl;
            break;
        }
    }

    std::cout << "\n--- Training Finished ---" << std::endl;
    if (epoch_count >= max_epochs && best_cost >= target_cost_threshold) {
        std::cout << "Max epochs (" << max_epochs << ") reached." << std::endl;
    }
    std::cout << "Final best cost: " << best_cost << " after " << epoch_count << " epochs and " << generation_count << " generations." << std::endl;

    std::cout << "\n--- Demonstrating Best Model Found ---" << std::endl;
    model_demonstrate(best_model, N_evolution_steps, target_image.width, target_image.height, "final_best_");

    std::cout << "\nProgram finished." << std::endl;
    return 0;
}
*/