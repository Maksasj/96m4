#pragma once

#include <cstddef>
#include <vector>
#include <functional>

#include "scalar.hpp"

namespace m964 {
    template<typename T, std::size_t Width, std::size_t Height>
    struct Layer {
        T* values;

        std::size_t width = Width; 
        std::size_t height = Height;

        explicit Layer() {
            values = new T[Width * Height];
        }

        Layer(const Layer& other) : width(other.width), height(other.height) {
            values = new T[Width * Height];
            std::copy(other.values, other.values + (Width * Height), values);
        }

        Layer& operator=(const Layer& other) {
            if (this != &other)
                std::copy(other.values, other.values + (Width * Height), values);
            
            return *this;
        }

        Layer(Layer&& other) noexcept : values(other.values) { 
            other.values = nullptr;
        }

        Layer& operator=(Layer&& other) noexcept {
            if (this != &other) {
                delete[] values;
                values = other.values;
                other.values = nullptr;
            }

            return *this;
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
}