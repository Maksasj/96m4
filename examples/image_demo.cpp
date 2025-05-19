#include <iostream>

#include <chrono>
#include <algorithm>
#include <string>
#include <random>
#include <execution>
#include <mutex>

#include "96m4.h"
#include "utils.hpp"

#include "stb_image.h"

#include "games/pong.hpp"

using namespace m964;

auto model_cost(Model &model, const size_t& steps, const std::vector<std::vector<float>>& image) -> float {
    model.reset_states();
    model.get_old_state().fill([]() {
        return rand_float(0.0f, 1.0f);
    });

    auto cost = 0.0f;

    for (std::int32_t t = 0; t < steps; ++t) {
        model.simulate_step();

        const auto sample = model.get_new_state();

        for(int i = 0; i < image.size(); ++i) {
            auto& row = image[i];

            for (int j = 0; j < row.size(); ++j) {
                auto& value = row[j];

                const auto t = (sample(i, j) - value);
                cost += t * t;
            }
        }
    }

    return cost / static_cast<float>(steps);
}

auto model_demonstrate(Model& model, const size_t& steps) -> void {
    model.reset_states();
    model.get_old_state().fill([]() {
        return rand_float(0.0f, 1.0f);
    });

    for (std::int32_t t = 0; t < steps; ++t) {
        model.simulate_step();

        export_state_as_image("state.png", model.get_new_state());
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    const auto sample = model.get_new_state();
    export_state_as_image("result.png", sample);
}

auto load_image(const std::string& filename) -> std::vector<std::vector<float>>  {
    int w, h, comp;
    auto raw = (unsigned int*) stbi_load(filename.c_str(), &w, &h, &comp, STBI_rgb_alpha);

    auto image = std::vector<std::vector<float>> {};

    for (int i = 0; i < w; ++i) {
        auto column = std::vector<float> {};

        for (int j = 0; j < h; ++j) {
            auto pixel = (raw[i + j * w] & 0x00000000ff) / 255.0f;
            column.push_back(pixel);
        }

        image.push_back(column);
    }

    stbi_image_free(raw);

    return image;
}

auto main() -> std::int32_t {
    auto image = load_image("test_image.png");

    auto steps = 10;
    auto best_mutex = std::mutex {};
    auto best = Model(32, 32);
    auto found_best = false;

    best.weights.fill([]() {
        return Kernel().fill(rand_float(-1.0, 1.0f));
    });

    auto best_cost = model_cost(best, steps, image);

    auto generation = 1;
    auto epoch = 0;

    while (true) {
        const auto mutation_rate = 1.0f / static_cast<float>(generation);

        auto models = std::vector<Model>{};
        models.reserve(100);

        for (auto i = 0; i < 100; ++i) {
            auto model = best;

            model.weights.apply(KernelOffset {
                -1.0f * mutation_rate, 1.0f * mutation_rate
            });

            models.push_back(model);
        }

        auto executor = ParallelExecutor();

        executor.execute(models.begin(), models.end(), [&](auto &model) {
            auto cost = model_cost(model, steps, image);

            best_mutex.lock();
            if (cost < best_cost) {
                best_cost = cost;
                best = model;
                found_best = true;
            }
            best_mutex.unlock();
        });


        if (found_best) {
            ++generation;
            found_best = false;
        }

        ++epoch;
        std::cout << "Epoch " << epoch << " (" <<  epoch * 1000 << ") with generation " << generation << " with best cost " << best_cost <<  "\n";

        if (best_cost < 10.0f)
            break;
    }

    std::cout << "Training is finished !\n";

    while (1)
        model_demonstrate(best, steps);

    return 0;
}
