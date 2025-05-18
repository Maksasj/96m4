#include <iostream>
#include <string>
#include <random>
#include <thread>
#include <chrono>

#include "96m4.h"
#include "utils.hpp"

auto main() -> std::int32_t {
    using namespace m964;

    auto a = Model(128u, 128u);
    
    a.states[0].fill([](const auto& x, const auto& y) {
        std::ignore = x;
        std::ignore = y;
        return m964::rand_float(-1.0, 1.0f);
    });
    
    a.weights.fill([](const auto& x, const auto& y) {
        std::ignore = x;
        std::ignore = y;
        return Kernel { -0.296, 0.304, -0.637, -0.226, -0.936, -0.051, 0.547, -0.034, 0.323}; 
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
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    return 0;
}