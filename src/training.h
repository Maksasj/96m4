#pragma once

#include <iostream>
#include <vector>
#include <limits>
#include <chrono>
#include <string>
#include <random>
#include <execution>
#include <mutex>
#include <string>
#include <iomanip> // Required for std::setfill and std::setw
#include <sstream> // Required for std::ostringstream

#include "model.h"
#include "parallel_executor.h"

namespace m964 {
    struct GeneticAlgorithmTrainingParameters {
        size_t model_width = 8;
        size_t model_height = 8;
        size_t n_evolution_steps = 16;
        int population_size = 100;
        float initial_mutation_strength = 0.1f;
        float target_cost_threshold = 0.5f;
        int max_epochs = 100000;
        int print_interval_epochs = 20;
    };

    std::string formatMilliseconds(long long milliseconds);
    double calculate_new_average(double old_average, int old_count, double new_entry);

    auto genetic_algorithm_training_hyper(std::function<float(Model&)> model_cost_callback, GeneticAlgorithmTrainingParameters parameters) -> Model;
}
