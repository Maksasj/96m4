#include <iostream>
#include <thread>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <string>
#include <random>

#include "96m4.h"
#include "stb_image_write.h"

class PongGame {
    public:
        int screen[16][32];

    private:
        int SCREEN_WIDTH;
        int SCREEN_HEIGHT;
        

        
        int PADDLE_WIDTH;
        int PADDLE_HEIGHT;
        int paddle_x;
        int paddle_y;
        
        int BALL_SIZE;
        int ball_x;
        int ball_y;
        int ball_vx;
        int ball_vy;

    private:
        void render_square(int x, int y, int size, int color) {
            for (int dy = 0; dy < size; dy++) {
                for (int dx = 0; dx < size; dx++) {
                    int drawX = x + dx;
                    int drawY = y + dy;
                    if (drawX >= 0 && drawX < SCREEN_WIDTH && drawY >= 0 && drawY < SCREEN_HEIGHT) {
                        screen[drawY][drawX] = color;
                    }
                }
            }
        }
        
        void render_rectangle(int x, int y, int width, int height, int color) {
            for (int dy = 0; dy < height; dy++) {
                for (int dx = 0; dx < width; dx++) {
                    int drawX = x + dx;
                    int drawY = y + dy;
        
                    if (drawX >= 0 && drawX < SCREEN_WIDTH && drawY >= 0 && drawY < SCREEN_HEIGHT) {
                        screen[drawY][drawX] = color;
                    }
                }
            }
        }
        
        void update_paddle() {
            int paddle_center = paddle_x + PADDLE_WIDTH / 2;
            if (ball_x > paddle_center)
                paddle_x++;
            else if (ball_x < paddle_center)
                paddle_x--;

            paddle_x = std::max(0, std::min(paddle_x, SCREEN_WIDTH - PADDLE_WIDTH));
        }

    public:
        void paddle_left() {
            paddle_x++;
            paddle_x = std::max(0, std::min(paddle_x, SCREEN_WIDTH - PADDLE_WIDTH));
        }

        void paddle_right() {
            paddle_x--;
            paddle_x = std::max(0, std::min(paddle_x, SCREEN_WIDTH - PADDLE_WIDTH));
        }

        PongGame() {
            SCREEN_WIDTH = 32;
            SCREEN_HEIGHT = 16;

            PADDLE_WIDTH = 8;
            PADDLE_HEIGHT = 1;
            paddle_x = SCREEN_WIDTH / 2 - PADDLE_WIDTH / 2;
            paddle_y = SCREEN_HEIGHT - 2;
            
            BALL_SIZE = 1;
            ball_x = SCREEN_WIDTH / 2;
            ball_y = SCREEN_HEIGHT / 2;
            ball_vx = 1;
            ball_vy = -1;

            clear_buffer();
        }
    
        bool game_over = false;
    
        auto is_game_over() -> bool {
            return game_over;
        }

        void simulate_frame() {
            ball_x += ball_vx;
            ball_y += ball_vy;
    
            if (ball_x <= 0 || ball_x >= SCREEN_WIDTH - BALL_SIZE)
                ball_vx = -ball_vx;
    
            if (ball_y <= 0)
                ball_vy = -ball_vy;
    
            update_paddle();
    
            if (ball_y >= paddle_y - BALL_SIZE && ball_x >= paddle_x && ball_x < paddle_x + PADDLE_WIDTH) {
                ball_vy = -ball_vy;
            }
    
            if (ball_y > (SCREEN_HEIGHT - 2)) {
                game_over = true;
            }
    
            clear_buffer();
            render_square(ball_x, ball_y, BALL_SIZE, 0xffffffff);
            render_rectangle(paddle_x, paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT, 0xffffffff);
            // display_buffer();
        }

        void clear_buffer() {
            for (int y = 0; y < SCREEN_HEIGHT; y++)
                for (int x = 0; x < SCREEN_WIDTH; x++)
                    screen[y][x] = 0;
        }
        
        void display_buffer() {
            stbi_write_jpg("game.png", SCREEN_WIDTH, SCREEN_HEIGHT, 4, screen, SCREEN_WIDTH * sizeof(int));
        }
};

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

template<typename T>
struct PlainValue {
    const T value;

    auto operator()(const auto& x, const auto& y) -> T {
        std::ignore = x;
        std::ignore = y;

        return value;
    }
};

template<typename T>
struct ClampValue {
    const T min;
    const T max;

    auto operator()(auto& value) -> void {
        if(value > max) value = max;
        if(value < min) value = min;
    }
};

template<typename T>
struct SinValue {
    const T min;
    const T max;

    auto operator()(auto& value) -> void {
        value = std::sin(value);
    }
};

#define EULER_NUMBER 2.71828
#define EULER_NUMBER_F 2.71828182846
#define EULER_NUMBER_L 2.71828182845904523536

template<typename T>
struct SigmoidValue {
    const T min;
    const T max;

    auto operator()(auto& value) -> void {
        value = (1 / (1 + std::pow(EULER_NUMBER, -value)));
    }
};

template<typename T>
struct ReluValue {
    const T min;
    const T max;

    auto operator()(auto& value) -> void {
        if(value < 0) value = 0;
    }
};

template<typename T>
struct KernelOffset {
    const T min;
    const T max;

    auto operator()(auto& value) {
        for(int j = 0; j < 9; ++j)
            value.values[j] += rand_float(min,max);
    }
};

auto main() -> int {
    using namespace m964;

    auto best = Model<float, Kernel3<float>, 32u, 16u>();

    best.weights.fill([](const auto& x, const auto& y) {
        std::ignore = x;
        std::ignore = y;
        return Kernel3<float>().fill(rand_float(-1.0, 1.0f));
    });
    
    int e = 0;
    int current_score = 0;
    bool first = true;
    while(1) {
        std::vector<Model<float, Kernel3<float>, 32u, 16u>> models;
    
        best.states[0].fill(PlainValue<float>{ 0.0f });
        best.states[1].fill(PlainValue<float>{ 0.0f });

        if(first) {
            for(int m = 0; m < 100; ++m) {
                models.push_back(Model<float, Kernel3<float>, 32u, 16u>());
                auto& model = models[models.size() - 1];

                model.weights.fill([](const auto& x, const auto& y) {
                    std::ignore = x;
                    std::ignore = y;
                    return Kernel3<float>().fill(rand_float(-1.0, 1.0f));
                });

                first = false;
            }
        } else {
            for(int m = 0; m < 100; ++m) {
                models.push_back(best);
                auto& model = models[models.size() - 1];
                auto& weights = model.weights; 
    
                weights.apply(KernelOffset<float>{ -1.0, 1.0f });
            }
        }

        std::cout << "Epoch " << e << "\n";
        ++e;

        for(auto& model : models) {
            PongGame game;

            int i = 0;
            int score = 0;
            while(!game.is_game_over()) {
                game.simulate_frame();

                auto o = i % 2;
                auto n = (i + 1) % 2;
        
                auto& old_state = model.states[o]; 
                auto& new_state = model.states[n]; 
                auto& weights = model.weights; 

                for(int i = 0; i < 32; ++i) {
                    for(int j = 0; j < 16; ++j) {
                        if(game.screen[j][i])
                            old_state(i, j) = 1.0f;
                        else
                            old_state(i, j) = 0.0f;
                    }
                }

                for(int t = 0; t < 12; ++t) {
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

                if(new_state(16, 8) < 0.5f) game.paddle_left();
                if(new_state(16, 8) > 0.5f) game.paddle_right();
                
                ++score;

                if(score > 300) {
                    std::cout << "Hit score limit !\n";
                    break;
                }
            }

            if(score > current_score) {
                current_score = score;
                best = model;

                std::cout << "New best with score " << score << "\n"; 
            }
        }

        if(current_score > 300) {
            best.states[0].fill(PlainValue<float>{ 0.0f });
            best.states[1].fill(PlainValue<float>{ 0.0f });

            PongGame game;

            int i = 0;
            while(!game.is_game_over()) {
                game.simulate_frame();

                auto o = i % 2;
                auto n = (i + 1) % 2;
        
                auto& old_state = best.states[o]; 
                auto& new_state = best.states[n]; 
                auto& weights = best.weights; 

                for(int i = 0; i < 32; ++i) {
                    for(int j = 0; j < 16; ++j) {
                        if(game.screen[j][i])
                            old_state(i, j) = 1.0f;
                        else
                            old_state(i, j) = 0.0f;
                    }
                }

                for(int t = 0; t < 12; ++t) {
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

                if(new_state(16, 8) < 0.5f)
                    game.paddle_left();

                if(new_state(16, 8) > 0.5f)
                    game.paddle_right();

                game.display_buffer();
            }
        }
    }

    return 0;
}
