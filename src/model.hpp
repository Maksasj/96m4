#pragma once

#include <cstddef>

#include "layer.hpp"
#include "scalar.hpp"

namespace m964 {
    template<typename C, typename K, std::size_t Width, std::size_t Height> requires Scalar<C>
    struct Model {
        Layer<C, Width, Height> states[2];
        Layer<K, Width, Height> weights;

        std::size_t width = Width;
        std::size_t height = Height;
    };

    template<typename C, typename K, std::size_t Width, std::size_t Height> requires Scalar<C>
    auto calculate_state(Layer<C, Width, Height>& new_state, const Layer<C, Width, Height>& state, const Layer<K, Width, Height>& weights) {
        for(size_t x = 1; x < Width - 1; ++x) {
            for(size_t y = 1; y < Height - 1; ++y) {
                auto value = state(x, y) * weights(x, y)(1, 1);
                value += state(x, y + 1) * weights(x, y + 1)(1, 2);
                value += state(x, y - 1) * weights(x, y - 1)(1, 0);
                value += state(x + 1, y) * weights(x + 1, y )(2, 1);
                value += state(x - 1, y) * weights(x - 1, y )(0, 1);

                value += state(x + 1, y + 1) * weights(x + 1, y + 1)(2, 2);
                value += state(x - 1, y - 1) * weights(x - 1, y - 1)(0, 0);

                value += state(x + 1, y - 1) * weights(x + 1, y - 1)(2, 0);
                value += state(x - 1, y + 1) * weights(x - 1, y + 1)(0, 2);

                new_state(x, y) = value;
            }
        }
    }
}