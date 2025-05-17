#include "model.h"

namespace m964 {
    Model::Model(
        const std::size_t& width, 
        const std::size_t& height
    ) : width(width),
        height(height),
        states{ Layer(width, height), Layer(width, height) },
        weights(width, height)
    {

    }

    auto Model::fill_states(const float& value) ->void {
        states[0].fill(value);
        states[1].fill(value);
    }

    auto calculate_state(Layer& new_state, const Layer& state, const KernelLayer& weights) -> void {
        const auto width = new_state.get_width();
        const auto height = new_state.get_height();

        for(size_t x = 1; x < width - 1; ++x) {
            for(size_t y = 1; y < height - 1; ++y) {
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