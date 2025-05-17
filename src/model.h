#pragma once

#include <cstddef>

#include "layer.h"
#include "kernel_layer.h"

namespace m964 {
    class Model {
        public:
            std::size_t width;
            std::size_t height;

            Layer states[2];
            KernelLayer weights;
      
        public:
            Model(const std::size_t& width, const std::size_t& height);

            auto fill_states(const float& value) -> void;
    };

    auto calculate_state(Layer& new_state, const Layer& state, const KernelLayer& weights) -> void;
}