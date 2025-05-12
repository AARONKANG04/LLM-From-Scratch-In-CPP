#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <unordered_set>

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    Tensor(const std::vector<float>& data, 
           const std::vector<int>& shape, 
           bool requires_grad = true);

    void backward();

    std::vector<float> data;
    std::vector<float> grad;
    std::vector<int> shape;
    std::function<void()> backward_fn = nullptr;
    std::vector<std::shared_ptr<Tensor>> prev;
    bool requires_grad;

private:
    void build_topo(std::shared_ptr<Tensor> v, 
                    std::vector<std::shared_ptr<Tensor>>& topo, 
                    std::unordered_set<Tensor*>& visited);
};

std::shared_ptr<Tensor> tensor(const std::vector<float>& data, 
                               const std::vector<int>& shape, 
                               bool requires_grad = true);

std::shared_ptr<Tensor> tensor_normal(const std::vector<int>& shape, 
                                      bool requires_grad = true, 
                                      float mu = 0.0f, 
                                      float std = 1.0f);