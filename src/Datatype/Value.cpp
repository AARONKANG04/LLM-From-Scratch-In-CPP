#include <Datatype/Value.hpp>

Value::Value(float value, bool requires_grad) 
    : value(value), grad(0.0f), requires_grad(requires_grad), backward_fn([](){}) {}

void Value::backward() {
    if (!requires_grad) return;

    std::vector<std::shared_ptr<Value>> topo;
    std::unordered_set<Value*> visited;
    build_topo(shared_from_this(), topo, visited);

    grad = 1.0f;
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        if ((*it)->backward_fn) (*it)->backward_fn();
    }
}

void Value::build_topo(std::shared_ptr<Value> v, 
                       std::vector<std::shared_ptr<Value>>& topo, 
                       std::unordered_set<Value*>& visited) {
    if (visited.count(v.get())) return;
    visited.insert(v.get());
    for (const auto& child : v->prev) {
        build_topo(child, topo, visited);
    }
    topo.push_back(v);
}

std::shared_ptr<Value> value(float data, bool requires_grad) {
    return std::make_shared<Value>(data, requires_grad);
}