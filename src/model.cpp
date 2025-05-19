#include "model.h"

namespace m964 {
    Model::Model(
        const std::size_t& width, 
        const std::size_t& height
    ) : width(width),
        height(height),
        weights(width, height)
    {

        states.emplace_back(width, height);
        states.emplace_back(width, height);

        reset_states();

    }

    auto Model::reset_states() -> void {
        old_state = 0;
        new_state = 1;

        fill_states(0.0f);
    }

    auto Model::fill_states(const float& value) ->void {
        states[old_state].fill(value);
        states[new_state].fill(value);
    }

    auto Model::simulate_step() -> void {
        auto& o_state = get_old_state();
        auto& n_state = get_new_state();

        calculate_state(n_state, o_state, weights);

        // n_state.apply(NormalizeValue());
        n_state.apply(SigmoidValue{});

        std::swap(old_state, new_state);
    }

    auto Model::get_new_state() -> Layer& {
        return states[new_state];
    }

    auto Model::get_old_state() -> Layer& {
        return states[old_state];
    }

    auto calculate_state(Layer& new_state, const Layer& state, const KernelLayer& weights) -> void {
        const auto width = new_state.get_width();
        const auto height = new_state.get_height();

        const auto width_m = width - 1;
        const auto height_m = height - 1;

        { // top left
            const auto& kernel = weights(0, 0);

            auto value = state(0, 0) * kernel(1, 1);
            value += state(0, 0 + 1) * kernel(1, 2);
            value += state(0 + 1, 0) * kernel(2, 1);
            value += state(0 + 1, 0 + 1) * kernel(2, 2);
            new_state(0, 0) = value;
        }

        { // top right
            const auto& kernel = weights(width_m, 0);

            auto value = state(width_m, 0) * kernel(1, 1);
            value += state(width_m, 0 + 1) * kernel(1, 2);
            value += state(width_m - 1, 0) * kernel(0, 1);
            value += state(width_m - 1, 0 + 1) * kernel(0, 2);

            new_state(width_m, 0) = value;
        }

        { // bottom left
            const auto& kernel = weights(0, height_m);

            auto value = state(0, height_m) * kernel(1, 1);
            value += state(0, height_m - 1) * kernel(1, 0);
            value += state(0 + 1, height_m) * kernel(2, 1);
            value += state(0 + 1, height_m - 1) * kernel(2, 0);

            new_state(0, height_m) = value;
        }

        { // bottom right
            const auto& kernel = weights(width_m, height_m);

            auto value = state(width_m, height_m) * kernel(1, 1);
            value += state(width_m, height_m - 1) * kernel(1, 0);
            value += state(width_m - 1, height_m) * kernel(0, 1);
            value += state(width_m - 1, height_m - 1) * kernel(0, 0);

            new_state(width_m, height_m) = value;
        }

        for(int x = 1; x < width_m; ++x) {
            {
                const auto& kernel = weights(x, 0);

                auto value = state(x, 0) * kernel(1, 1);
                value += state(x, 0 + 1) * kernel(1, 2);
                value += state(x + 1, 0) * kernel(2, 1);
                value += state(x - 1, 0) * kernel(0, 1);
                value += state(x + 1, 0 + 1) * kernel(2, 2);
                value += state(x - 1, 0 + 1) * kernel(0, 2);

                new_state(x, 0) = value;
            }

            {
                const auto& kernel = weights(x, height_m);

                auto value = state(x, height_m) * kernel(1, 1);
                value += state(x, height_m - 1) * kernel(1, 0);
                value += state(x + 1, height_m) * kernel(2, 1);
                value += state(x - 1, height_m) * kernel(0, 1);
                value += state(x - 1, height_m - 1) * kernel(0, 0);
                value += state(x + 1, height_m - 1) * kernel(2, 0);

                new_state(x, 0) = value;
            }
        }

        for(int y = 1; y < height_m; ++y) {
            {
                const auto& kernel = weights(0, y);

                auto value = state(0, y) * kernel(1, 1);
                value += state(0, y + 1) * kernel(1, 2);
                value += state(0, y - 1) * kernel(1, 0);
                value += state(0 + 1, y) * kernel(2, 1);
                value += state(0 + 1, y + 1) * kernel(2, 2);
                value += state(0 + 1, y - 1) * kernel(2, 0);

                new_state(0, y) = value;
            }
            {
                const auto& kernel = weights(width_m, y);

                auto value = state(width_m, y) * kernel(1, 1);
                value += state(width_m, y + 1) * kernel(1, 2);
                value += state(width_m, y - 1) * kernel(1, 0);
                value += state(width_m + 1, y) * kernel(2, 1);
                value += state(width_m + 1, y + 1) * kernel(2, 2);
                value += state(width_m + 1, y - 1) * kernel(2, 0);

                new_state(width_m, y) = value;
            }
        }

        for(int x = 1; x < width_m; ++x) {
            for(int y = 1; y < height_m; ++y) {
                const auto& kernel = weights(x, y);

                auto value = state(x, y) * kernel(1, 1);
                value += state(x, y + 1) * kernel(1, 2);
                value += state(x, y - 1) * kernel(1, 0);
                value += state(x + 1, y) * kernel(2, 1);
                value += state(x - 1, y) * kernel(0, 1);

                value += state(x + 1, y + 1) * kernel(2, 2);
                value += state(x - 1, y - 1) * kernel(0, 0);

                value += state(x + 1, y - 1) * kernel(2, 0);
                value += state(x - 1, y + 1) * kernel(0, 2);

                new_state(x, y) = value;
            }
        }
    }
}