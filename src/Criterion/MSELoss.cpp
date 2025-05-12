#include <Criterion/MSELoss.hpp>
#include <Datatype/Tensor_ops.hpp>

std::shared_ptr<Tensor> MSELoss::forward(std::shared_ptr<Tensor> preds, 
                                         std::shared_ptr<Tensor> targets) {
    return mse_loss(preds, targets);
}
