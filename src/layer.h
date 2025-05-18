#pragma once

#include <cstddef>
#include <vector>
#include <functional>

namespace m964 {
    class Layer {
        private:
            std::size_t width; 
            std::size_t height;

            std::vector<float> values;

        public:
            Layer(const std::size_t& width, const std::size_t& height);

            auto fill(const float& value) -> Layer&;
            auto fill(const std::function<float()>& lambda) -> Layer&;
            auto fill(const std::function<float(const std::size_t&, const std::size_t&)>& lambda) -> Layer&;

            auto apply(const std::function<void(float&)>& lambda) -> Layer&;

            auto get_width() const -> std::size_t;
            auto get_height() const -> std::size_t;

            auto operator()(const size_t& x, const size_t& y) -> float&;
            auto operator()(const size_t& x, const size_t& y) const -> const float&;
    };
}
