#include <Layer/Softmax.hpp>
#include <Datatype/Tensor_ops.hpp>

std::shared_ptr<Tensor> Softmax::forward(std::shared_ptr<Tensor> input) {
    return softmax_last_dim(input);
}

std::shared_ptr<Softmax> softmax() {
    return std::make_shared<Softmax>();
}