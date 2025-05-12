#pragma once

#include "Value.hpp"

std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b);
std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b);
std::shared_ptr<Value> sigmoid(const std::shared_ptr<Value>& a);
std::shared_ptr<Value> mse_loss(const std::shared_ptr<Value>& pred, const std::shared_ptr<Value>& label);