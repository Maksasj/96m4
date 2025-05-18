#pragma once

#include <cstddef>
#include <vector>
#include <functional>

#include "kernel.h"

namespace m964 {
    class KernelLayer {
        private:
            std::size_t width; 
            std::size_t height;

            std::vector<Kernel> values;
        
        public:
            explicit KernelLayer(const std::size_t& width, const std::size_t& height);

            auto fill(const Kernel& value) -> void;
            auto fill(const std::function<Kernel()>& lambda) -> void;
            auto fill(const std::function<Kernel(const std::size_t&, const std::size_t&)>& lambda) -> void;

            auto apply(const std::function<void(Kernel&)>& lambda) -> KernelLayer&;

            [[nodiscard]] auto get_width() const -> std::size_t;
            [[nodiscard]] auto get_height() const -> std::size_t;

            auto operator()(const size_t& x, const size_t& y) -> Kernel&;
            auto operator()(const size_t& x, const size_t& y) const -> const Kernel&;
    };

}
