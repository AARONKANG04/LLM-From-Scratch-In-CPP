#pragma once

#include "Tensor.hpp"

std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& A, const std::shared_ptr<Tensor>& B);
std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& A, const std::shared_ptr<Tensor>& B);
std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor>& A, const std::shared_ptr<Tensor>& B);
std::shared_ptr<Tensor> relu(const std::shared_ptr<Tensor>& A);
std::shared_ptr<Tensor> sigmoid(const std::shared_ptr<Tensor>& A);
std::shared_ptr<Tensor> mse_loss(const std::shared_ptr<Tensor>& preds, const std::shared_ptr<Tensor>& labels);
std::shared_ptr<Tensor> embedding(const std::shared_ptr<Tensor>& indices, const std::shared_ptr<Tensor>& embedding_matrix);
std::shared_ptr<Tensor> cross_entropy_loss(const std::shared_ptr<Tensor>& preds, const std::shared_ptr<Tensor>& targets);
std::shared_ptr<Tensor> softmax_last_dim(const std::shared_ptr<Tensor>& A);
std::shared_ptr<Tensor> transpose_last2(const std::shared_ptr<Tensor>& A);
std::shared_ptr<Tensor> scale(const std::shared_ptr<Tensor>& A, float factor);
std::shared_ptr<Tensor> concat_last_dim(std::vector<std::shared_ptr<Tensor>> data);