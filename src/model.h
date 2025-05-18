#pragma once

#include <cstddef>

#include "layer.h"
#include "kernel_layer.h"
#include "utils.h"

namespace m964 {
    class Model {
        public:
            std::size_t width;
            std::size_t height;

            Layer states[2];
            KernelLayer weights;

            std::size_t old_state;
            std::size_t new_state;

        public:
            Model(const std::size_t& width, const std::size_t& height);

            auto reset_states() -> void;
            auto fill_states(const float& value) -> void;
            auto simulate_step() -> void;

            auto get_new_state() -> Layer&;
            auto get_old_state() -> Layer&;
    };

    auto calculate_state(Layer& new_state, const Layer& state, const KernelLayer& weights) -> void;
}