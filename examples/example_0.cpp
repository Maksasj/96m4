#include <iostream>
#include <string>
#include <random>

#include "96m4.h"

#include "stb_image_write.h"


template<typename C, std::size_t Width, std::size_t Height> requires m964::Scalar<C>
auto export_state_as_image(const std::string& file_name, const m964::Layer<C, Width, Height>& state) -> void {
    auto buffer = std::vector<int> {};
    buffer.resize(Width * Height);

    for(size_t x = 0; x < Width; ++x) {
        for(size_t y = 0; y < Height; ++y) {
            auto value = state(x, y);

            unsigned char r = (unsigned char) (value * 255.0f);
            unsigned char g = (unsigned char) (value * 255.0f);
            unsigned char b = (unsigned char) (value * 255.0f);

            buffer[x + y*Width] = (255 << 24) | (b << 16) | (g << 8) | (r);
        }
    }

    stbi_write_jpg(file_name.c_str(), Width, Height, 4, buffer.data(), Width * sizeof(int));
}

auto rand_float(const float& min, const float& max) -> float {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);
    return dist(gen);
}

auto main() -> int {
    using namespace m964;

    auto a = Model<float, Kernel3<float>, 1000u, 1000u>();
    
    a.states[0].fill([](const auto& x, const auto& y) {
        std::ignore = x;
        std::ignore = y;
        return rand_float(-1.0, 1.0f);
    });

    a.weights.fill([](const auto& x, const auto& y) {
        std::ignore = x;
        std::ignore = y;
        // return Kernel3<float>().fill(rand_float(-1.0, 1.0f));
        
        return Kernel3<float> { -0.296, 0.304, -0.637, -0.226, -0.936, -0.051, 0.547, -0.034, 0.323}; 
    });

    for(auto i = 0; i < 1000000; ++i) {
        const auto o = i % 2;
        const auto n = (i + 1) % 2;

        auto& old_state = a.states[o]; 
        auto& new_state = a.states[n]; 
        auto& weights = a.weights; 

        calculate_state(new_state, old_state, weights);

        new_state.apply([](auto& value) {
            if(value > 1.0f) value = 1.0f;
            if(value < 0.0f) value = 0.0f;
        });

        export_state_as_image("state.png", new_state);
    }

    return 0;
}