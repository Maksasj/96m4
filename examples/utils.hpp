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

auto export_state_as_image(const std::string& file_name, const m964::Layer& state) -> void {
    const auto width = state.get_width();
    const auto height = state.get_height();

	auto buffer = std::vector<std::int32_t> {};
    buffer.resize(width * height);

    for(size_t x = 0; x < width; ++x) {
        for(size_t y = 0; y < height; ++y) {
            auto value = state(x, y);
            buffer[x + y*width] = hsl_to_rgb((1 - value) * 255, 0.5f, 1.0f);
        }
    }

    stbi_write_jpg(file_name.c_str(), width, height, 4, buffer.data(), width * sizeof(std::int32_t));
}

std::uint32_t direction_to_rgb_magnitude_grouping(float* direction) {
    float magR_sq = direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2];
    float magG_sq = direction[3] * direction[3] + direction[4] * direction[4] + direction[5] * direction[5];
    float magB_sq = direction[6] * direction[6] + direction[7] * direction[7] + direction[8] * direction[8];

    float magR = sqrtf(magR_sq);
    float magG = sqrtf(magG_sq);
    float magB = sqrtf(magB_sq);

    const float max_sub_magnitude = sqrtf(3.0f);

    float r_norm = magR / max_sub_magnitude;
    float g_norm = magG / max_sub_magnitude;
    float b_norm = magB / max_sub_magnitude;

	auto r = static_cast<unsigned char>(std::clamp(r_norm * 255.0f, 0.0f, 255.0f));
	auto g = static_cast<unsigned char>(std::clamp(g_norm * 255.0f, 0.0f, 255.0f));
	auto b = static_cast<unsigned char>(std::clamp(b_norm * 255.0f, 0.0f, 255.0f));

    return (255 << 24) | (b << 16) | (g << 8) | (r);
}

auto export_state_as_image(const std::string& file_name, const m964::KernelLayer& state) -> void {
    const auto width = state.get_width();
    const auto height = state.get_height();

	auto buffer = std::vector<std::int32_t> {};
    buffer.resize(width * height);

    for(size_t x = 0; x < width; ++x) {
        for(size_t y = 0; y < height; ++y) {
            auto value = state(x, y);
            buffer[x + y*width] = direction_to_rgb_magnitude_grouping(value.values);
        }
    }

    stbi_write_jpg(file_name.c_str(), width, height, 4, buffer.data(), width * sizeof(std::int32_t));
}