#include "training.h"

namespace m964 {
    std::string formatMilliseconds(long long milliseconds) {
        long long totalSeconds = milliseconds / 1000;
        long long ms = milliseconds % 1000;
        long long ss = totalSeconds % 60;
        long long totalMinutes = totalSeconds / 60;
        long long mm = totalMinutes % 60;
        long long hh = totalMinutes / 60;

        std::ostringstream oss;
        oss << std::setfill('0') << std::setw(2) << hh << ":"
            << std::setfill('0') << std::setw(2) << mm << ":"
            << std::setfill('0') << std::setw(2) << ss << ":"
            << std::setfill('0') << std::setw(3) << ms;

        return oss.str();
    }

    double calculate_new_average(double old_average, int old_count, double new_entry) {
        // Basic validation: The count of numbers should not be negative.
        // If it's zero and a new entry is added, the new average is simply the new entry.
        if (old_count < 0) {
            std::cerr << "Error: The count of numbers cannot be negative." << std::endl;
            return 0.0; // Indicate an error
        }

        // If the old count was 0, it means there were no previous numbers.
        // In this case, the new average is simply the new entry itself.
        if (old_count == 0) {
            return new_entry;
        }

        // Calculate the sum of the old numbers.
        // The sum is derived from the old average and the old count: Sum = Average * Count
        double old_sum = old_average * old_count;

        // Calculate the new sum by adding the new entry to the old sum.
        double new_sum = old_sum + new_entry;

        // Calculate the new count of numbers.
        // The new count is simply the old count plus one (for the new entry).
        int new_count = old_count + 1;

        // Calculate the new average.
        // The new average is the new sum divided by the new count: New Average = New Sum / New Count
        double new_average = new_sum / new_count;

        return new_average;
    }

    auto genetic_algorithm_training_hyper(std::function<float(Model&)> model_cost_callback, GeneticAlgorithmTrainingParameters parameters) -> Model {
        const auto n_evolution_steps = parameters.n_evolution_steps;
        const auto population_size = parameters.population_size;
        const auto initial_mutation_strength = parameters.initial_mutation_strength;
        const auto target_cost_threshold = parameters.target_cost_threshold;
        const auto max_epochs = parameters.max_epochs;
        const auto print_interval_epochs = parameters.print_interval_epochs;

        auto best_model = Model(parameters.model_width, parameters.model_height);

        std::cout << "Initialized base model with dimensions: " << 4 << "x" << 4 << std::endl;

        best_model.bias_layer.fill([&]() { // Assuming rand_float can be used here effectively
            return rand_float(-initial_mutation_strength, initial_mutation_strength); // Example: smaller initial weight range
        });

        best_model.weights.fill([&]() { // Assuming rand_float can be used here effectively
            return Kernel().fill(rand_float(-initial_mutation_strength, initial_mutation_strength)); // Example: smaller initial weight range
        });

        auto best_cost = model_cost_callback(best_model);
        auto prev_cost = best_cost;

        std::cout << "Initial model cost: " << best_cost << std::endl;

        auto best_mutex = std::mutex{};
        bool found_new_best_this_epoch = false;
        long long generation_count = 1;
        long long epoch_count = 0;
        float epoch_avg_time = 0.0f;

        std::cout << "\n--- Starting Training ---" << std::endl;
        std::cout << "N_evolution_steps: " << n_evolution_steps
                  << ", Population: " << population_size
                  << ", Target Cost: < " << target_cost_threshold
                  << ", Max Epochs: " << max_epochs << std::endl;

        while (epoch_count < max_epochs) {
            auto epoch_start_time = std::chrono::high_resolution_clock::now();

            std::vector<Model> current_population(population_size);

            const float current_mutation_strength = initial_mutation_strength / std::sqrt(static_cast<float>(generation_count));

            for (int i = 0; i < population_size; ++i) {
                current_population[i] = best_model;

                current_population[i].bias_layer.apply([&](auto& value) {
                    value += rand_float(-1.0f, 1.0f) * current_mutation_strength;
                });

                current_population[i].weights.apply(KernelOffset{
                    -current_mutation_strength, current_mutation_strength
                });
            }

            found_new_best_this_epoch = false;
            ParallelExecutor executor; // Assuming this is your parallel execution utility


            executor.execute(current_population.begin(), current_population.end(), [&](Model &candidate_model) {
                float candidate_cost = model_cost_callback(candidate_model);

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

            const double costChangePerEpoch = prev_cost - best_cost;
            const double totalCostReductionNeeded = best_cost - target_cost_threshold;
            auto predicted_epochs = static_cast<std::size_t>(std::ceil(totalCostReductionNeeded / costChangePerEpoch));

            epoch_avg_time = calculate_new_average(epoch_avg_time, epoch_count, epoch_duration_ms);

            if (found_new_best_this_epoch) {
                ++generation_count;

                printf("Epoch %lld | Gen %lld | New Best Cost: %.6f | Predicted : %lld | Mut.Strength: %.4f | Epoch Time: %lldms | Estimated epoch max time: [ %s ] | *Improvement!*\n",
                       epoch_count, generation_count, best_cost, predicted_epochs, current_mutation_strength, epoch_duration_ms, formatMilliseconds((max_epochs - epoch_count) * epoch_avg_time).c_str());
            } else {
                if (epoch_count % print_interval_epochs == 0) {
                    printf("Epoch %lld | Gen %lld | New Best Cost: %.6f | Predicted : %lld | Mut.Strength: %.4f | Epoch Time: %lldms | Estimated epoch max time: [ %s ]\n",
                           epoch_count, generation_count, best_cost, predicted_epochs, current_mutation_strength, epoch_duration_ms, formatMilliseconds((max_epochs - epoch_count) * epoch_avg_time).c_str());
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


        return best_model;
    }
}