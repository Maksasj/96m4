#pragma once

#include <cstddef>
#include <vector>
#include <functional>

#include "scalar.hpp"

namespace m964 {
    template<typename T, std::size_t Width, std::size_t Height>
    struct Layer {
        T* values;

        const std::size_t width = Width; 
        const std::size_t height = Height;

        explicit Layer() {
            values = new T[Width * Height];
        }

        ~Layer() {
            delete [] values;
        }        

        auto fill(const T& value) -> Layer<T, Width, Height>& {
            for(std::size_t x = 0; x < width; ++x) {
                for(std::size_t y = 0; y < height; ++y) {
                    values[x + y*width] = value;
                }
            }

            return *this;
        }

        auto fill(const std::function<T(const std::size_t&, const std::size_t&)>& lambda) -> Layer<T, Width, Height>& {
            for(std::size_t x = 0; x < width; ++x) {
                for(std::size_t y = 0; y < height; ++y) {
                    values[x + y*width] = lambda(x, y);
                }
            }

            return *this;
        }

        auto apply(const std::function<void(T&)>& lambda) -> Layer<T, Width, Height>& {
            for(std::size_t x = 0; x < Width; ++x) {
                for(std::size_t y = 0; y < Height; ++y) {
                    lambda(values[x + y*Width]);
                }
            }

            return *this;
        }

        inline auto operator()(const size_t& x, const size_t& y) -> T& {
            return values[x + y*width];
        }

        inline auto operator()(const size_t& x, const size_t& y) const -> const T& {
            return values[x + y*width];
        }
    };

    template<typename C, typename K, std::size_t Width, std::size_t Height> requires Scalar<C>
    struct Model {
        Layer<C, Width, Height> states[2];
        Layer<K, Width, Height> weights;

        const std::size_t width = Width; 
        const std::size_t height = Height; 
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