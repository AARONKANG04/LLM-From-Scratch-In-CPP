#include <Layer/ReLU.hpp>
#include <Datatype/Tensor_ops.hpp>

std::shared_ptr<Tensor> ReLU::forward(std::shared_ptr<Tensor> input) {
    return relu(input);
}

std::shared_ptr<ReLU> relu() {
    return std::make_shared<ReLU>();
}

