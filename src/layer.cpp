#include "layer.h"

namespace m964 {
    Layer::Layer(const std::size_t& width, const std::size_t& height) : width(width), height(height) {
        values.resize(width * height);
    }

    auto Layer::fill(const float& value) -> Layer& {
        for(std::size_t x = 0; x < width; ++x)
            for(std::size_t y = 0; y < height; ++y)
                values[x + y*width] = value;

        return *this;
    }

    auto Layer::fill(const std::function<float()>& lambda) -> Layer& {
        for(std::size_t x = 0; x < width; ++x)
            for(std::size_t y = 0; y < height; ++y)
                values[x + y*width] = lambda();

        return *this;
    }

    auto Layer::fill(const std::function<float(const std::size_t&, const std::size_t&)>& lambda) -> Layer& {
        for(std::size_t x = 0; x < width; ++x)
            for(std::size_t y = 0; y < height; ++y)
                values[x + y*width] = lambda(x, y);

        return *this;
    }

    auto Layer::apply(const std::function<void(float&)>& lambda) -> Layer& {
        for(std::size_t x = 0; x < width; ++x)
            for(std::size_t y = 0; y < height; ++y)
                lambda(values[x + y*width]);

        return *this;
    }

    auto Layer::get_width() const -> std::size_t {
        return width;
    }

    auto Layer::get_height() const -> std::size_t {
        return height;
    }

    auto Layer::operator()(const size_t& x, const size_t& y) -> float& {
        return values[x + y*width];
    }

    auto Layer::operator()(const size_t& x, const size_t& y) const -> const float& {
        return values[x + y*width];
    }
}