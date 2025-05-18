#include "kernel.h"

namespace m964 {
    auto Kernel::fill(const float& value) -> Kernel& {
        for(std::size_t x = 0; x < 3; ++x) {
            for(std::size_t y = 0; y < 3; ++y) {
                values[x + y*3] = value;
            }
        }

        return *this;
    }

    auto Kernel::fill(const std::function<float(const std::size_t&, const std::size_t&)>& lambda) -> Kernel& {
        for(std::size_t x = 0; x < 3; ++x) {
            for(std::size_t y = 0; y < 3; ++y) {
                values[x + y*3] = lambda(x, y);
            }
        }

        return *this;
    }

    auto Kernel::operator()(const size_t& x, const size_t& y) -> float& {
        return values[x + y*3];
    }

    auto Kernel::operator()(const size_t& x, const size_t& y) const -> const float& {
        return values[x + y*3];
    }
}