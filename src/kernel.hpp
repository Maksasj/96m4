#pragma once

#include "scalar.hpp"

namespace m964 {
    template<typename T, std::size_t Width, std::size_t Height> requires Scalar<T>
    struct Kernel {
        T values[Width * Height];

        auto fill(const T& value) -> Kernel<T, Width, Height>& {
            for(std::size_t x = 0; x < Width; ++x) {
                for(std::size_t y = 0; y < Height; ++y) {
                    values[x + y*Width] = value;
                }
            }

            return *this;
        }

        auto fill(const std::function<T(const std::size_t&, const std::size_t&)>& lambda) -> Kernel<T, Width, Height>& {
            for(std::size_t x = 0; x < Width; ++x) {
                for(std::size_t y = 0; y < Height; ++y) {
                    values[x + y*Width] = lambda(x, y);
                }
            }

            return *this;
        }

        inline auto operator()(const size_t& x, const size_t& y) -> T& {
            return values[x + y*Width];
        }

        inline auto operator()(const size_t& x, const size_t& y) const -> const T& {
            return values[x + y*Width];
        }
    };

    template<typename T>
    using Kernel3 = Kernel<T, 3, 3>;
}