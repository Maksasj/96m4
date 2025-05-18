#include "kernel_layer.h"

namespace m964 {
    KernelLayer::KernelLayer(const std::size_t& width, const std::size_t& height) : width(width), height(height) {
        values.resize(width * height);
    }

    auto KernelLayer::fill(const Kernel& value) -> void  {
        for (auto& v : values)
            v = value;
    }

    auto KernelLayer::fill(const std::function<Kernel()>& lambda) -> void {
        for (auto& value : values)
            value = lambda();
    }

    auto KernelLayer::fill(const std::function<Kernel(const std::size_t&, const std::size_t&)>& lambda) -> void {
        for(std::size_t x = 0; x < width; ++x)
            for(std::size_t y = 0; y < height; ++y)
                values[x + y*width] = lambda(x, y);
    }

    auto KernelLayer::apply(const std::function<void(Kernel&)>& lambda) -> KernelLayer& {
        for(std::size_t x = 0; x < width; ++x)
            for(std::size_t y = 0; y < height; ++y)
                lambda(values[x + y*width]);

        return *this;
    }

    auto KernelLayer::get_width() const -> std::size_t {
        return width;
    }

    auto KernelLayer::get_height() const -> std::size_t {
        return height;
    }

    auto KernelLayer::operator()(const size_t& x, const size_t& y) -> Kernel& {
        return values[x + y*width];
    }

    auto KernelLayer::operator()(const size_t& x, const size_t& y) const -> const Kernel& {
        return values[x + y*width];
    }
}