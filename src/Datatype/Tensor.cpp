#include <Datatype/Tensor.hpp>

#include <random>

Tensor::Tensor(const std::vector<float>& data, 
               const std::vector<int>& shape, 
               bool requires_grad) 
               : data(data), shape(shape), requires_grad(requires_grad) {
    grad.resize(data.size(), 0.0f);
}   

void Tensor::backward() {
    std::vector<std::shared_ptr<Tensor>> topo;
    std::unordered_set<Tensor*> visited;
    build_topo(shared_from_this(), topo, visited);

    grad.assign(data.size(), 1.0f);
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        if ((*it)->backward_fn) (*it)->backward_fn();
    }
}

void Tensor::build_topo(std::shared_ptr<Tensor> v, 
                        std::vector<std::shared_ptr<Tensor>>& topo, 
                        std::unordered_set<Tensor*>& visited) {
    if (visited.count(v.get())) return;
    visited.insert(v.get());
    for (const auto& child : v->prev) {
        build_topo(child, topo, visited);
    }
    topo.push_back(v);
}

std::shared_ptr<Tensor> tensor(const std::vector<float>& data, 
                               const std::vector<int>& shape, 
                               bool requires_grad) {
    return std::make_shared<Tensor>(data, shape, requires_grad);
}

std::shared_ptr<Tensor> tensor_normal(const std::vector<int>& shape, 
                                      bool requires_grad, 
                                      float mu, 
                                      float std) {
    size_t total_size = 1;
    for (int dim : shape) total_size *= dim;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(mu, std);

    std::vector<float> data(total_size);

    for (size_t i = 0; i < total_size; ++i) {
        data[i] = dist(gen);
    }

    return std::make_shared<Tensor>(data, shape, requires_grad);
}