#include "model.h"
#include <stdexcept> // For runtime_error, if you choose to use exceptions

namespace m964 {
    Model::Model(
    ) : width(DEFAULT_MODEL_STATE_DIM_X),
        height(DEFAULT_MODEL_STATE_DIM_Y),
        bias_layer(width, height),
        weights(DEFAULT_MODEL_STATE_DIM_X, DEFAULT_MODEL_STATE_DIM_Y),
        old_state(0),
        new_state(0) // KernelLayer constructor called with model dimensions
    {
            states.emplace_back(width, height);
            states.emplace_back(width, height);
            reset_states(); // This will also call fill_states
    }

    Model::Model(
        const std::size_t& width,
        const std::size_t& height
    ) : width(width),
        height(height),
        bias_layer(width, height),
        weights(width, height),
        old_state(0),
        new_state(0) // KernelLayer constructor called with model dimensions
    {
        states.emplace_back(width, height);
        states.emplace_back(width, height);
        reset_states(); // This will also call fill_states
    }

    auto Model::reset_states() -> void {
        old_state = 0;
        new_state = 1;
        // Ensure states are not empty before trying to fill.
        // The constructor should guarantee states are populated.
        if (states.size() >= 2) { // Or check against old_state and new_state bounds
             fill_states(0.0f);
        } else {
            // This case should ideally not happen if constructor logic is sound.
            std::cerr << "Warning [Model::reset_states]: States vector not properly initialized." << std::endl;
        }
    }

    auto Model::fill_states(const float& value) -> void {
        // Check if indices are valid for the states vector
        if (old_state < states.size() && new_state < states.size()) {
            states[old_state].fill(value);
            states[new_state].fill(value);
        } else {
            std::cerr << "Warning [Model::fill_states]: State indices out of bounds." << std::endl;
        }
    }

    auto Model::simulate_step() -> void {
        // Ensure states are valid before proceeding
        if (old_state >= states.size() || new_state >= states.size()) {
            std::cerr << "Error [Model::simulate_step]: State indices invalid. Cannot simulate." << std::endl;
            return;
        }
        auto& biases = bias_layer;
        auto& o_state = get_old_state();
        auto& n_state = get_new_state();

        calculate_state(n_state, o_state, weights);

        // n_state.apply(NormalizeValue());
        // n_state.apply(SigmoidValue{});
        n_state.apply(ReluValue());

        std::swap(old_state, new_state);
    }

    auto Model::simulate_step_with_biases() -> void {
        // Ensure states are valid before proceeding
        if (old_state >= states.size() || new_state >= states.size()) {
            std::cerr << "Error [Model::simulate_step]: State indices invalid. Cannot simulate." << std::endl;
            return;
        }
        auto& biases = bias_layer;
        auto& o_state = get_old_state();
        auto& n_state = get_new_state();

        calculate_state_with_biases(n_state, o_state, biases, weights);

        // n_state.apply(NormalizeValue());
        // n_state.apply(SigmoidValue{});
        n_state.apply(ReluValue());

        std::swap(old_state, new_state);
    }

    auto Model::get_new_state() -> Layer& {
        // Add bounds check for safety, though ideally indices are always valid.
        if (new_state >= states.size()) {
            std::cerr << "Error [Model::get_new_state]: new_state index out of bounds!" << std::endl;
            // Consider throwing an exception or returning a reference to a static 'dummy' layer
            // For now, this might lead to a crash if not handled, which is loud but indicates a problem.
            // This situation implies a logic error elsewhere.
            throw std::out_of_range("new_state index is out of bounds for states vector");
        }
        return states[new_state];
    }

    auto Model::get_old_state() -> Layer& {
        if (old_state >= states.size()) {
            std::cerr << "Error [Model::get_old_state]: old_state index out of bounds!" << std::endl;
            throw std::out_of_range("old_state index is out of bounds for states vector");
        }
        return states[old_state];
    }

    auto Model::save_to_file(const std::string& filename) const -> bool {
        std::ofstream file(filename, std::ios::binary | std::ios::trunc);
        if (!file.is_open()) {
            std::cerr << "Error [Model::save_to_file]: Could not open file for saving: " << filename << std::endl;
            return false;
        }

        file.write(reinterpret_cast<const char*>(&this->width), sizeof(this->width));
        file.write(reinterpret_cast<const char*>(&this->height), sizeof(this->height));

        for (std::size_t r = 0; r < this->height; ++r) {
            for (std::size_t c = 0; c < this->width; ++c) {
                const Kernel& current_kernel = this->weights(c, r);
                for (std::size_t k_r = 0; k_r < DEFAULT_KERNEL_DIM_Y; ++k_r) {
                    for (std::size_t k_c = 0; k_c < DEFAULT_KERNEL_DIM_X; ++k_c) {
                        float val = current_kernel(k_c, k_r);
                        file.write(reinterpret_cast<const char*>(&val), sizeof(val));
                    }
                }
            }
        }

        if (!file.good()) {
            std::cerr << "Error [Model::save_to_file]: Failed to write all data to file: " << filename << std::endl;
            file.close();
            return false;
        }
        file.close();
        std::cout << "Model saved successfully to " << filename << std::endl;
        return true;
    }

    // Static method implementation
    auto Model::load_from_file(const std::string& filename) -> std::optional<Model> {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error [Model::load_from_file]: Could not open file for loading: " << filename << std::endl;
            return std::nullopt;
        }

        std::size_t loaded_width;
        std::size_t loaded_height;

        file.read(reinterpret_cast<char*>(&loaded_width), sizeof(loaded_width));
        file.read(reinterpret_cast<char*>(&loaded_height), sizeof(loaded_height));

        if (!file.good() || loaded_width == 0 || loaded_height == 0) {
            std::cerr << "Error [Model::load_from_file]: Invalid dimensions or failed to read dimensions from: " << filename << std::endl;
            file.close();
            return std::nullopt;
        }

        // Create a new Model instance with the loaded dimensions.
        // This will also initialize its states and weights structures.
        Model loaded_model(loaded_width, loaded_height);

        // Now, populate the weights of the newly created loaded_model
        for (std::size_t r = 0; r < loaded_model.height; ++r) {
            for (std::size_t c = 0; c < loaded_model.width; ++c) {
                // `loaded_model.weights(c, r)` must return `Kernel&`.
                Kernel& current_kernel = loaded_model.weights(c, r);
                for (std::size_t k_r = 0; k_r < DEFAULT_KERNEL_DIM_Y; ++k_r) {
                    for (std::size_t k_c = 0; k_c < DEFAULT_KERNEL_DIM_X; ++k_c) {
                        float val;
                        file.read(reinterpret_cast<char*>(&val), sizeof(val));
                        if (!file.good()) {
                             std::cerr << "Error [Model::load_from_file]: Failed to read kernel data for kernel at ("
                                       << c << "," << r << ") from: " << filename << std::endl;
                             file.close();
                             return std::nullopt;
                        }
                        // Assumes Kernel::operator()(k_c, k_r) returns float& for assignment
                        current_kernel(k_c, k_r) = val;
                    }
                }
            }
        }

        if (!file.eof() && !file.good()) {
            std::cerr << "Error [Model::load_from_file]: Problem reading file " << filename << " after expected kernel data." << std::endl;
            file.close();
            return std::nullopt;
        }

        file.close();
        std::cout << "Model loaded successfully from " << filename
                  << ". Dimensions: " << loaded_model.width << "x" << loaded_model.height << std::endl;

        return loaded_model; // Return the fully constructed and populated model
    }

    // calculate_state function remains unchanged
    auto calculate_state(Layer& new_state, const Layer& state,  const KernelLayer& weights) -> void {
        const auto width = new_state.get_width();
        const auto height = new_state.get_height();

        const auto width_m = width - 1;
        const auto height_m = height - 1;

        { // top left
            const auto& kernel = weights(0, 0);
            auto value = state(0, 0) * kernel(1, 1);
            value += state(0, 0 + 1) * kernel(1, 2);
            value += state(0 + 1, 0) * kernel(2, 1);
            value += state(0 + 1, 0 + 1) * kernel(2, 2);
            new_state(0, 0) = value;
        }

        { // top right
            const auto& kernel = weights(width_m, 0);
            auto value = state(width_m, 0) * kernel(1, 1);
            value += state(width_m, 0 + 1) * kernel(1, 2);
            value += state(width_m - 1, 0) * kernel(0, 1);
            value += state(width_m - 1, 0 + 1) * kernel(0, 2);
            new_state(width_m, 0) = value;
        }

        { // bottom left
            const auto& kernel = weights(0, height_m);
            auto value = state(0, height_m) * kernel(1, 1);
            value += state(0, height_m - 1) * kernel(1, 0);
            value += state(0 + 1, height_m) * kernel(2, 1);
            value += state(0 + 1, height_m - 1) * kernel(2, 0);
            new_state(0, height_m) = value;
        }

        { // bottom right
            const auto& kernel = weights(width_m, height_m);
            auto value = state(width_m, height_m) * kernel(1, 1);
            value += state(width_m, height_m - 1) * kernel(1, 0);
            value += state(width_m - 1, height_m) * kernel(0, 1);
            value += state(width_m - 1, height_m - 1) * kernel(0, 0);
            new_state(width_m, height_m) = value;
        }

        for(std::size_t x = 1; x < width_m; ++x) {
            { // Top edge
                const auto& kernel = weights(x, 0);
                auto value = state(x, 0) * kernel(1, 1);
                value += state(x, 0 + 1) * kernel(1, 2);
                value += state(x + 1, 0) * kernel(2, 1);
                value += state(x - 1, 0) * kernel(0, 1);
                value += state(x + 1, 0 + 1) * kernel(2, 2);
                value += state(x - 1, 0 + 1) * kernel(0, 2);
                new_state(x, 0) = value;
            }
            { // Bottom edge
                const auto& kernel = weights(x, height_m);
                auto value = state(x, height_m) * kernel(1, 1);
                value += state(x, height_m - 1) * kernel(1, 0);
                value += state(x + 1, height_m) * kernel(2, 1);
                value += state(x - 1, height_m) * kernel(0, 1);
                value += state(x + 1, height_m - 1) * kernel(2, 0);
                value += state(x - 1, height_m - 1) * kernel(0, 0);
                new_state(x, height_m) = value;
            }
        }

        for(std::size_t y = 1; y < height_m; ++y) {
            { // Left edge
                const auto& kernel = weights(0, y);
                auto value = state(0, y) * kernel(1, 1);
                value += state(0, y + 1) * kernel(1, 2);
                value += state(0, y - 1) * kernel(1, 0);
                value += state(0 + 1, y) * kernel(2, 1);
                value += state(0 + 1, y + 1) * kernel(2, 2);
                value += state(0 + 1, y - 1) * kernel(2, 0);
                new_state(0, y) = value;
            }
            { // Right edge
                const auto& kernel = weights(width_m, y);
                auto value = state(width_m, y) * kernel(1, 1);
                value += state(width_m, y + 1) * kernel(1, 2);
                value += state(width_m, y - 1) * kernel(1, 0);
                value += state(width_m - 1, y) * kernel(0, 1);
                value += state(width_m - 1, y + 1) * kernel(0, 2);
                value += state(width_m - 1, y - 1) * kernel(0, 0);
                new_state(width_m, y) = value;
            }
        }

        for(std::size_t x = 1; x < width_m; ++x) {
            for(std::size_t y = 1; y < height_m; ++y) {
                const auto& kernel = weights(x, y);
                auto value = state(x, y) * kernel(1, 1);
                value += state(x, y + 1) * kernel(1, 2);
                value += state(x, y - 1) * kernel(1, 0);
                value += state(x + 1, y) * kernel(2, 1);
                value += state(x - 1, y) * kernel(0, 1);
                value += state(x + 1, y + 1) * kernel(2, 2);
                value += state(x - 1, y - 1) * kernel(0, 0);
                value += state(x + 1, y - 1) * kernel(2, 0);
                value += state(x - 1, y + 1) * kernel(0, 2);
                new_state(x, y) = value;
            }
        }
    }

        // calculate_state function remains unchanged
    auto calculate_state_with_biases(Layer& new_state, const Layer& state,  const Layer& biases, const KernelLayer& weights) -> void {
        const auto width = new_state.get_width();
        const auto height = new_state.get_height();

        calculate_state(new_state, state, weights);

        for (std::size_t x = 1; x < width; ++x)
            for (std::size_t y = 1; y < height; ++y)
                new_state(x, y) += biases(x, y);
    }
}