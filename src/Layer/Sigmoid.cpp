#include <Layer/Sigmoid.hpp>
#include <Datatype/Tensor_ops.hpp>

std::shared_ptr<Tensor> Sigmoid::forward(std::shared_ptr<Tensor> input) {
    return sigmoid(input);
}

std::shared_ptr<Sigmoid> sigmoid() {
    return std::make_shared<Sigmoid>();
}

