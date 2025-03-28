#pragma once

#include <random>

#include "96m4.h"
#include "stb_image_write.h"

float hue_to_rgb(float v1, float v2, float vH) {
	if (vH < 0)
		vH += 1;

	if (vH > 1)
		vH -= 1;

	if ((6 * vH) < 1)
		return (v1 + (v2 - v1) * 6 * vH);

	if ((2 * vH) < 1)
		return v2;

	if ((3 * vH) < 2)
		return (v1 + (v2 - v1) * ((2.0f / 3) - vH) * 6);

	return v1;
}

std::uint32_t hsl_to_rgb(float H, float L, float S) {
    std::uint32_t rgb;

	if (S == 0) {
        const auto v = (std::uint8_t)(L * 255);
        rgb = (255 << 24) | (v << 16) | (v << 8) | (v);
	} else {
		float v1, v2;
		float hue = (float)H / 360;

		v2 = (L < 0.5) ? (L * (1 + S)) : ((L + S) - (L * S));
		v1 = 2 * L - v2;

		auto r = (std::uint8_t)(255 * hue_to_rgb(v1, v2, hue + (1.0f / 3)));
		auto g = (std::uint8_t)(255 * hue_to_rgb(v1, v2, hue));
		auto b = (std::uint8_t)(255 * hue_to_rgb(v1, v2, hue - (1.0f / 3)));

        rgb = (255 << 24) | (b << 16) | (g << 8) | (r);
	}

	return rgb;
}

template<typename C, std::size_t Width, std::size_t Height> requires m964::Scalar<C>
auto export_state_as_image(const std::string& file_name, const m964::Layer<C, Width, Height>& state) -> void {
    auto buffer = std::vector<std::int32_t> {};
    buffer.resize(Width * Height);

    for(size_t x = 0; x < Width; ++x) {
        for(size_t y = 0; y < Height; ++y) {
            auto value = state(x, y);
            buffer[x + y*Width] = hsl_to_rgb((1 - value) * 255, 0.5f, 1.0f);
        }
    }

    stbi_write_jpg(file_name.c_str(), Width, Height, 4, buffer.data(), Width * sizeof(std::int32_t));
}

