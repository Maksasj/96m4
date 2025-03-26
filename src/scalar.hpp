#pragma once

namespace m964 {
    template<typename T>
    concept Addable = requires (T a, T b)
    {
        a + b;
    };

    template<typename T>
    concept Multipliable = requires (T a, T b)
    {
        a * b;
    };

    template<typename T>
    concept Scalar = Addable<T> && Multipliable<T>;
}