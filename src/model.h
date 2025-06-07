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

            auto reset_states() -> void;
            auto fill_states(const float& value) -> void;
            auto simulate_step() -> void;
            auto simulate_step_with_biases() -> void;

            auto get_new_state() -> Layer&;
            auto get_old_state() -> Layer&;
    };

    auto calculate_state(Layer& new_state, const Layer& state, const KernelLayer& weights) -> void;
    auto calculate_state_with_biases(Layer& new_state, const Layer& state,  const Layer& biases, const KernelLayer& weights) -> void;
}