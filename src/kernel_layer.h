#pragma once

#include <cstddef>
#include <vector>
#include <functional>

#include "kernel.hpp"

namespace m964 {
    class KernelLayer {
        private:
            std::size_t width; 
            std::size_t height;

            Kernel3<float>* values;
        
        public:
            explicit KernelLayer(const std::size_t& width, const std::size_t& height);
            ~KernelLayer();

            KernelLayer(const KernelLayer& other);
            KernelLayer& operator=(const KernelLayer& other);
            KernelLayer(KernelLayer&& other) noexcept;
            KernelLayer& operator=(KernelLayer&& other) noexcept;

            auto fill(const Kernel3<float>& value) -> KernelLayer&;
            auto fill(const std::function<Kernel3<float>()>& lambda) -> KernelLayer&;
            auto fill(const std::function<Kernel3<float>(const std::size_t&, const std::size_t&)>& lambda) -> KernelLayer&;

            auto apply(const std::function<void(Kernel3<float>&)>& lambda) -> KernelLayer&;

            auto get_width() const -> std::size_t;
            auto get_height() const -> std::size_t;

            auto operator()(const size_t& x, const size_t& y) -> Kernel3<float>&;
            auto operator()(const size_t& x, const size_t& y) const -> const Kernel3<float>&;
    };
}
