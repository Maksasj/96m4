#pragma once

class PongGame {
    public:
        std::int32_t screen[16][32];
    std::int32_t paddle_x;
    std::int32_t paddle_y;

    std::int32_t BALL_SIZE;
    std::int32_t ball_x;
    std::int32_t ball_y;
    std::int32_t ball_vx;
    std::int32_t ball_vy;

    private:
        std::int32_t SCREEN_WIDTH;
        std::int32_t SCREEN_HEIGHT;
        
        std::int32_t PADDLE_WIDTH;
        std::int32_t PADDLE_HEIGHT;


    private:
        void render_square(std::int32_t x, std::int32_t y, std::int32_t size, std::int32_t color) {
            for (std::int32_t dy = 0; dy < size; dy++) {
                for (std::int32_t dx = 0; dx < size; dx++) {
                    std::int32_t drawX = x + dx;
                    std::int32_t drawY = y + dy;
                    if (drawX >= 0 && drawX < SCREEN_WIDTH && drawY >= 0 && drawY < SCREEN_HEIGHT) {
                        screen[drawY][drawX] = color;
                    }
                }
            }
        }
        
        void render_rectangle(std::int32_t x, std::int32_t y, std::int32_t width, std::int32_t height, std::int32_t color) {
            for (std::int32_t dy = 0; dy < height; dy++) {
                for (std::int32_t dx = 0; dx < width; dx++) {
                    std::int32_t drawX = x + dx;
                    std::int32_t drawY = y + dy;
        
                    if (drawX >= 0 && drawX < SCREEN_WIDTH && drawY >= 0 && drawY < SCREEN_HEIGHT) {
                        screen[drawY][drawX] = color;
                    }
                }
            }
        }

    public:
        auto paddle_prediction() const -> float {
            // const auto paddle = (paddle_x + PADDLE_WIDTH/2);
            const auto paddle = paddle_x;

            if (ball_x > paddle)
                return 1.0f;

            if (ball_x < paddle)
                return 0.0f;

            return 0.5f;
        }

        void paddle_left() {
            paddle_x--;
            paddle_x = std::max(0, std::min(paddle_x, SCREEN_WIDTH - PADDLE_WIDTH));
        }

        void paddle_right() {
            paddle_x++;
            paddle_x = std::max(0, std::min(paddle_x, SCREEN_WIDTH - PADDLE_WIDTH));
        }

        auto get_paddle_position() -> std::pair<int, int> {
            return std::make_pair(paddle_x, paddle_y);
        }

        auto get_ball_position() -> std::pair<int, int> {
            return std::make_pair(ball_x, ball_y);
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
            // uodate ball
            ball_x += ball_vx;
            ball_y += ball_vy;
    
            if (ball_x <= 0 || ball_x >= SCREEN_WIDTH - BALL_SIZE)
                ball_vx = -ball_vx;
    
            if (ball_y <= 0)
                ball_vy = -ball_vy;

            if (ball_y >= paddle_y - BALL_SIZE && ball_x >= paddle_x && ball_x < paddle_x + PADDLE_WIDTH) {
                ball_vy = -ball_vy;
            }
    
            if (ball_y > (SCREEN_HEIGHT - 2)) {
                game_over = true;
            }

            // render screen
            clear_buffer();
            render_square(ball_x, ball_y, BALL_SIZE, 0xffffffff);
            render_rectangle(paddle_x, paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT, 0xffffff00);
        }

        void clear_buffer() {
            for (std::int32_t y = 0; y < SCREEN_HEIGHT; y++)
                for (std::int32_t x = 0; x < SCREEN_WIDTH; x++)
                    screen[y][x] = 0;
        }
        
        void display_buffer() {
            stbi_write_jpg("game.png", SCREEN_WIDTH, SCREEN_HEIGHT, 4, screen, SCREEN_WIDTH * sizeof(std::int32_t));
        }
};