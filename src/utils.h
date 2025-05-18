#pragma once

#include <random>

#include "kernel.h"

#define EULER_NUMBER 2.71828
#define EULER_NUMBER_F 2.71828182846
#define EULER_NUMBER_L 2.71828182845904523536

namespace m964 {
    auto rand_int(const int& min, const int& max) -> int;

    auto rand_float(const float& min, const float& max) -> float;

    struct PlainValue {
        const float value;

        auto operator()(const float& x, const float& y) const -> float;
    };

    struct ClampValue {
        const float min;
        const float max;

        auto operator()(float& value) const -> void;
    };

    struct NormalizeValue : public ClampValue {
        NormalizeValue();
    };

    struct SinValue {
        const float min;
        const float max;

        auto operator()(float& value) const -> void;
    };

    struct SigmoidValue {
        const float min;
        const float max;

        auto operator()(float& value) const -> void;
    };

    struct ReluValue {
        const float min;
        const float max;

        auto operator()(float& value) const -> void;
    };

    struct KernelOffset {
        const float min;
        const float max;

        auto operator()(Kernel& value) const -> void;
    };
}

