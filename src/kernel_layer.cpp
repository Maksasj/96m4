#include "kernel_layer.h"

namespace m964 {
    KernelLayer::KernelLayer(const std::size_t& width, const std::size_t& height) : width(width), height(height) {
        values = new Kernel3<float>[width * height];
    }

    KernelLayer::KernelLayer(const KernelLayer& other) : width(other.width), height(other.height) {
        values = new Kernel3<float>[width * height];
        std::copy(other.values, other.values + (width * height), values);
    }

    KernelLayer& KernelLayer::operator=(const KernelLayer& other) {
        if (this != &other)
            std::copy(other.values, other.values + (width * height), values);
        
        return *this;
    }

    KernelLayer::KernelLayer(KernelLayer&& other) noexcept : width(width), height(height), values(other.values) { 
        other.values = nullptr;
    }

    KernelLayer& KernelLayer::operator=(KernelLayer&& other) noexcept {
        if (this != &other) {
            delete[] values;
            values = other.values;
            other.values = nullptr;
        }

        return *this;
    }

    KernelLayer::~KernelLayer() {
        delete [] values;
    }

    auto KernelLayer::fill(const Kernel3<float>& value) -> KernelLayer& {
        for(std::size_t x = 0; x < width; ++x)
            for(std::size_t y = 0; y < height; ++y)
                values[x + y*width] = value;

        return *this;
    }

    auto KernelLayer::fill(const std::function<Kernel3<float>()>& lambda) -> KernelLayer& {
        for(std::size_t x = 0; x < width; ++x)
            for(std::size_t y = 0; y < height; ++y)
                values[x + y*width] = lambda();

        return *this;
    }

    auto KernelLayer::fill(const std::function<Kernel3<float>(const std::size_t&, const std::size_t&)>& lambda) -> KernelLayer& {
        for(std::size_t x = 0; x < width; ++x)
            for(std::size_t y = 0; y < height; ++y)
                values[x + y*width] = lambda(x, y);

        return *this;
    }

    auto KernelLayer::apply(const std::function<void(Kernel3<float>&)>& lambda) -> KernelLayer& {
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

    auto KernelLayer::operator()(const size_t& x, const size_t& y) -> Kernel3<float>& {
        return values[x + y*width];
    }

    auto KernelLayer::operator()(const size_t& x, const size_t& y) const -> const Kernel3<float>& {
        return values[x + y*width];
    }
}