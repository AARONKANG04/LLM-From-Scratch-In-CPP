#include <Criterion/CrossEntropyLoss.hpp>
#include <Datatype/Tensor_ops.hpp>

std::shared_ptr<Tensor> CrossEntropyLoss::forward(const std::shared_ptr<Tensor>& preds, const std::shared_ptr<Tensor>& targets) {
    return cross_entropy_loss(preds, targets);
}

