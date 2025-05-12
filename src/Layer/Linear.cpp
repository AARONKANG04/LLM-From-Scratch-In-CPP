#include <Layer/Linear.hpp>
#include <Datatype/Tensor_ops.hpp>

#include <cmath>

Linear::Linear(int in_features, int out_features, bool use_bias) : use_bias(use_bias) {
    float sigma = std::sqrt(2.0f / float(in_features));
    weights = tensor_normal({in_features, out_features}, true, 0.0f, sigma);
    if (use_bias) {
        std::vector<float> zeros(out_features, 0.0f);
        biases = tensor(zeros, {1, out_features}, true);
    }
}

std::shared_ptr<Tensor> Linear::forward(std::shared_ptr<Tensor> input) {
    auto out = matmul(input, weights);
    if (use_bias) out = out + biases;
    return out;
}

std::vector<std::shared_ptr<Tensor>> Linear::parameters() {
    return use_bias ? std::vector{weights, biases} : std::vector{weights};
}

std::shared_ptr<Linear> linear(int in_features, int out_features, bool use_bias) {
    return std::make_shared<Linear>(in_features, out_features, use_bias);
}
