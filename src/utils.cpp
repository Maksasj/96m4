#include "utils.h"

namespace m964 {
    auto rand_int(const int& min, const int& max) -> int {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist(min, max);
        return dist(gen);
    }

    auto rand_float(const float& min, const float& max) -> float {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(min, max);
        return dist(gen);
    }

    auto PlainValue::operator()(const float& x, const float& y) const -> float {
        std::ignore = x;
        std::ignore = y;

        return value;
    }

    auto ClampValue::operator()(float& value) const -> void {
        if(value > max) value = max;
        if(value < min) value = min;
    }

    NormalizeValue::NormalizeValue() : ClampValue(0.0f, 1.0f) {

    }

    auto SinValue::operator()(float& value) const -> void {
        value = std::sin(value);
    }

    auto SigmoidValue::operator()(float& value) const -> void {
        value = 1.0f / (1.0f + std::pow(EULER_NUMBER, -value));
    }

    auto ReluValue::operator()(float& value) const -> void {
        if(value < 0)
            value = 0;
    }

    auto KernelOffset::operator()(Kernel& value) const -> void {
        for(auto& j : value.values)
            j += rand_float(min,max);
    }
}

