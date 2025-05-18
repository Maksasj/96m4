#pragma once

#include <functional>

namespace m964 {
    struct Kernel {
        float values[9];

        auto fill(const float& value) -> Kernel&;

        auto fill(const std::function<float(const std::size_t&, const std::size_t&)>& lambda) -> Kernel&;

        auto operator()(const size_t& x, const size_t& y) -> float&;
        auto operator()(const size_t& x, const size_t& y) const -> const float&;
    };
}