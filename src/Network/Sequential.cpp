#include <Network/Sequential.hpp>

void Sequential::add(const std::shared_ptr<Module>& layer) {
    layers.push_back(layer);
}

std::shared_ptr<Tensor> Sequential::forward(std::shared_ptr<Tensor> input) {
    for (const auto& layer : layers) {
        input = layer->forward(input);
    }
    return input;
}

std::vector<std::shared_ptr<Tensor>> Sequential::parameters() {
    std::vector<std::shared_ptr<Tensor>> params;
    for (const auto& layer : layers) {
        auto sub = layer->parameters();
        params.insert(params.end(), sub.begin(), sub.end());
    }
    return params;
}
