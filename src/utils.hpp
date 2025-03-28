#pragma once

namespace m964 {
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
            for(std::int32_t j = 0; j < 9; ++j)
                value.values[j] += m964::rand_float(min,max);
        }
    };

}

