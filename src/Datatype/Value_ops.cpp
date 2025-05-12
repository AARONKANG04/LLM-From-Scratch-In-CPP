#include <Datatype/Value_ops.hpp>

#include <cmath>

std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    bool requires_grad = a->requires_grad || b->requires_grad;
    auto s = std::make_shared<Value>(a->value + b->value, requires_grad);
    s->prev = {a, b};
    s->backward_fn = [a, b, s]() {
        if (a->requires_grad) a->grad += s->grad;
        if (b->requires_grad) b->grad += s->grad;
    };
    return s;
}

std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    bool requires_grad = a->requires_grad || b->requires_grad;
    auto p = std::make_shared<Value>(a->value * b->value, requires_grad);
    p->prev = {a, b};
    p->backward_fn = [a, b, p]() {
        if (a->requires_grad) a->grad += p->grad * b->value;
        if (b->requires_grad) b->grad += p->grad * a->value;
    };
    return p;
}

std::shared_ptr<Value> sigmoid(const std::shared_ptr<Value>& a) {
    float sig = 1.0f / (1.0f + std::exp(-a->value));
    auto s = std::make_shared<Value>(sig, a->requires_grad);
    s->prev = {a};
    s->backward_fn = [a, s]() {
        float sig_grad = s->value * (1.0f - s->value);
        if (a->requires_grad) a->grad += s->grad * sig_grad;
    };
    return s;
}

std::shared_ptr<Value> mse_loss(const std::shared_ptr<Value>& pred, const std::shared_ptr<Value>& label) {
    float diff = pred->value - label->value;
    auto l = std::make_shared<Value>(diff * diff, pred->requires_grad || label->requires_grad);
    l->prev = {pred, label};
    l->backward_fn = [pred, label, l]() {
        float diff = pred->value - label->value;
        if (pred->requires_grad) pred->grad += 2.0f * diff * l->grad;
        if (label->requires_grad) label->grad -= 2.0f * diff * l->grad;
    };
    return l;
}
