#include <iostream>
#include <thread>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <string>
#include <random>
#include <execution>

#include "96m4.h"
#include "utils.hpp"

#include "games/pong.hpp"

auto main() -> std::int32_t {
    using namespace m964;

    std::mutex best_mutex;
    auto best = Model<float, Kernel3<float>, 32u, 16u>();
    std::size_t best_score = 0;
    
    best.weights.fill([](const auto& x, const auto& y) {
        std::ignore = x;
        std::ignore = y;
        return Kernel3<float>().fill(m964::rand_float(-1.0, 1.0f));
    });
    
    std::size_t generation = 1;
    std::size_t epoch = 0;
 
    while(1) {
        best.states[0].fill(PlainValue<float>{ 0.0f });
        best.states[1].fill(PlainValue<float>{ 0.0f });

        auto models = std::vector<Model<float, Kernel3<float>, 32u, 16u>>(100);

        std::generate(models.begin(), models.end(), [&]() mutable { 
            Model<float, Kernel3<float>, 32u, 16u> model = best;
            model.weights.apply(KernelOffset<float>{ -1.0f / static_cast<float>(generation), 1.0f / static_cast<float>(generation) });
            return model;
        });

        std::for_each(std::execution::par, models.begin(), models.end(), [&](auto& model) {
            auto game = PongGame();

            std::size_t i = 0;
            std::size_t score = 0;

            while(!game.is_game_over()) {
                game.simulate_frame();

                auto o = i % 2;
                auto n = (i + 1) % 2;
        
                auto& old_state = model.states[o]; 
                auto& new_state = model.states[n]; 
                auto& weights = model.weights; 

                for(std::int32_t i = 0; i < 32; ++i) {
                    for(std::int32_t j = 0; j < 16; ++j) {
                        if(game.screen[j][i])
                            old_state(i, j) = 1.0f;
                        else
                            old_state(i, j) = 0.0f;
                    }
                }

                for(std::int32_t t = 0; t < 64; ++t) {
                    o = i % 2;
                    n = (i + 1) % 2;
            
                    old_state = model.states[o]; 
                    new_state = model.states[n]; 
                    weights = model.weights; 
                    
                    calculate_state(new_state, old_state, weights);
                    
                    // apply clamp
                    new_state.apply(ClampValue<float> { 0.0f, 1.0f });
                    new_state.apply(ReluValue<float> {});
                    ++i;
                }

                if(new_state(3, 3) < 0.5f) game.paddle_left();
                if(new_state(3, 3) > 0.5f) game.paddle_right();
                
                ++score;

                if(score > 10000)
                    break;
            }

            best_mutex.lock();
            if(score > best_score) {
                best_score = score;
                best = model;
                ++generation;
            }
            best_mutex.unlock();
        });
        
        ++epoch;
        std::cout << "Epoch " << epoch << ", " << "with best score " << best_score << "\n";

        if(best_score > 10000) {
            best.states[0].fill(PlainValue<float>{ 0.0f });
            best.states[1].fill(PlainValue<float>{ 0.0f });

            PongGame game;

            std::int32_t i = 0;
            while(!game.is_game_over()) {
                game.simulate_frame();

                auto o = i % 2;
                auto n = (i + 1) % 2;
        
                auto& old_state = best.states[o]; 
                auto& new_state = best.states[n]; 
                auto& weights = best.weights; 

                for(std::int32_t i = 0; i < 32; ++i) {
                    for(std::int32_t j = 0; j < 16; ++j) {
                        if(game.screen[j][i])
                            old_state(i, j) = 1.0f;
                        else
                            old_state(i, j) = 0.0f;
                    }
                }

                for(std::int32_t t = 0; t < 64; ++t) {
                    o = i % 2;
                    n = (i + 1) % 2;
            
                    old_state = best.states[o]; 
                    new_state = best.states[n]; 
                    weights = best.weights; 
                    
                    calculate_state(new_state, old_state, weights);
                    
                    // apply clamp
                    new_state.apply(ClampValue<float> { 0.0f, 1.0f });
                    new_state.apply(ReluValue<float> {});
                    
                    export_state_as_image("state.png", new_state);
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    ++i;
                }

                if(new_state(3, 3) < 0.5f) game.paddle_left();
                if(new_state(3, 3) > 0.5f) game.paddle_right();

                game.display_buffer();
            }
        }
    }

    return 0;
}
