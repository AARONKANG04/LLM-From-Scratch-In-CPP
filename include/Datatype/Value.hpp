#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <unordered_set>

class Value : public std::enable_shared_from_this<Value> {
public:
    Value(float value, bool requires_grad = true);
    void backward();
    
    float value;
    float grad;
    std::vector<std::shared_ptr<Value>> prev;
    std::function<void()> backward_fn = nullptr;
    bool requires_grad;

private:
    void build_topo(std::shared_ptr<Value> v, 
                    std::vector<std::shared_ptr<Value>>& topo, 
                    std::unordered_set<Value*>& visited);
};

std::shared_ptr<Value> value(const float data, bool requires_grad = true);