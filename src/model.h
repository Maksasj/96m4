#pragma once

#include <cstddef>
#include <vector>
#include <iostream>
#include <fstream>   // For file operations
#include <string>    // For filename
#include <optional>  // For std::optional

#include "layer.h"
#include "kernel_layer.h"
#include "utils.h"

namespace m964 {
    constexpr std::size_t DEFAULT_KERNEL_DIM_X = 3;
    constexpr std::size_t DEFAULT_KERNEL_DIM_Y = 3;

    constexpr std::size_t DEFAULT_MODEL_STATE_DIM_X = 8;
    constexpr std::size_t DEFAULT_MODEL_STATE_DIM_Y = 8;

    class Model {
    public:
        std::size_t width;
        std::size_t height;

        Layer bias_layer;
        std::vector<Layer> states;
        KernelLayer weights;

        std::size_t old_state;
        std::size_t new_state;

    public:
        Model();
        Model(const std::size_t& width, const std::size_t& height);
        // Add a default constructor if you want to be able to declare
        // Model m; and then m = Model::load_from_file(...).value_or(default_model);
        // Or if you might create an uninitialized model for other reasons.
        // For this static load, it's not strictly necessary if you always assign the result.
        // Model() : width(0), height(0), weights(0,0), old_state(0), new_state(1) {}

        auto reset_states() -> void;
        auto fill_states(const float& value) -> void;
        auto simulate_step() -> void;
        auto simulate_step_with_biases() -> void;

        auto get_new_state() -> Layer&;
        auto get_old_state() -> Layer&;

        // Save method remains an instance method
        [[nodiscard]] auto save_to_file(const std::string& filename) const -> bool;

        // Static load method
        static auto load_from_file(const std::string& filename) -> std::optional<Model>;
    };

    auto calculate_state(Layer& new_state, const Layer& state, const KernelLayer& weights) -> void;
    auto calculate_state_with_biases(Layer& new_state, const Layer& state,  const Layer& biases, const KernelLayer& weights) -> void;
}