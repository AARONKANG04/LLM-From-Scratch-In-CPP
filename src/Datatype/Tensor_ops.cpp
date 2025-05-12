#include <Datatype/Tensor.hpp>
#include <Datatype/Tensor_ops.hpp>

#include <omp.h>
#include <cassert>
#include <vector>
#include <algorithm>
#include <cmath>
#include <memory>
#include <iostream>
#include <arm_neon.h>

std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& A, 
                                  const std::shared_ptr<Tensor>& B) {
    bool requires_grad = A->requires_grad || B->requires_grad;

    const auto& shapeA = A->shape;
    const auto& shapeB = B->shape;
    assert(shapeA.size() >= shapeB.size());

    size_t offset = shapeA.size() - shapeB.size();
    for (size_t i = 0; i < shapeB.size(); ++i) {
        assert((shapeA[i + offset] == shapeB[i]) || (shapeB[i] == 1));
    }

    size_t total = A->data.size();
    std::vector<float> out_data(total);

    std::vector<size_t> b_strides(shapeB.size(), 1);
    for (int d = shapeB.size() - 2; d >= 0; --d) {
        b_strides[d] = b_strides[d + 1] * shapeB[d + 1];
    }

    #pragma omp parallel
    {
        std::vector<int> index(shapeA.size());
        #pragma omp for
        for (size_t idx = 0; idx < total; ++idx) {
            size_t tmp = idx;
            for (int d = int(shapeA.size()) - 1; d >= 0; --d) {
                index[d] = tmp % shapeA[d];
                tmp /= shapeA[d];
            }

            size_t b_idx = 0;
            for (int d = 0; d < shapeB.size(); ++d) {
                int ix = index[d + offset];
                if (shapeB[d] == 1) ix = 0;
                b_idx += ix * b_strides[d];
            }

            out_data[idx] = A->data[idx] + B->data[b_idx];
        }
    }

    auto out = std::make_shared<Tensor>(out_data, shapeA, requires_grad);
    out->prev = {A, B};
    out->backward_fn = [A, B, out, shapeA, shapeB, offset, b_strides]() {
        if (A->requires_grad) {
            #pragma omp parallel for
            for (size_t i = 0; i < A->grad.size(); ++i) {
                A->grad[i] += out->grad[i];
            }
        }

        if (B->requires_grad) {
            #pragma omp parallel
            {
                std::vector<int> index(shapeA.size());
                #pragma omp for
                for (size_t idx = 0; idx < out->grad.size(); ++idx) {
                    size_t tmp = idx;
                    for (int d = int(shapeA.size()) - 1; d >= 0; --d) {
                        index[d] = tmp % shapeA[d];
                        tmp /= shapeA[d];
                    }

                    size_t b_idx = 0;
                    for (int d = 0; d < shapeB.size(); ++d) {
                        int ix = index[d + offset];
                        if (shapeB[d] == 1) ix = 0;
                        b_idx += ix * b_strides[d];
                    }

                    #pragma omp atomic
                    B->grad[b_idx] += out->grad[idx];
                }
            }
        }
    };
    return out;
}

std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& A,
                                  const std::shared_ptr<Tensor>& B) {
    bool requires_grad = A->requires_grad || B->requires_grad;

    const auto& shapeA = A->shape;
    const auto& shapeB = B->shape;
    assert(shapeA.size() >= shapeB.size());

    size_t offset = shapeA.size() - shapeB.size();
    for (size_t i = 0; i < shapeB.size(); ++i) {
        assert((shapeA[i + offset] == shapeB[i]) || (shapeB[i] == 1));
    }

    size_t total = A->data.size();
    std::vector<float> out_data(total);

    std::vector<size_t> b_strides(shapeB.size(), 1);
    for (int d = shapeB.size() - 2; d >= 0; --d) {
        b_strides[d] = b_strides[d + 1] * shapeB[d + 1];
    }

    #pragma omp parallel
    {
        std::vector<int> index(shapeA.size()); 
        #pragma omp for
        for (size_t idx = 0; idx < total; ++idx) {
            size_t tmp = idx;
            for (int d = int(shapeA.size()) - 1; d >= 0; --d) {
                index[d] = tmp % shapeA[d];
                tmp /= shapeA[d];
            }

            size_t b_idx = 0;
            for (int d = 0; d < shapeB.size(); ++d) {
                int ix = index[d + offset];
                if (shapeB[d] == 1) ix = 0;
                b_idx += ix * b_strides[d];
            }

            out_data[idx] = A->data[idx] - B->data[b_idx];
        }
    }

    auto out = std::make_shared<Tensor>(out_data, shapeA, requires_grad);
    out->prev = {A, B};
    out->backward_fn = [A, B, out, shapeA, shapeB, offset, b_strides]() {
        if (A->requires_grad) {
            #pragma omp parallel for
            for (size_t i = 0; i < A->grad.size(); ++i) {
                A->grad[i] += out->grad[i];
            }
        }

        if (B->requires_grad) {
            #pragma omp parallel
            {
                std::vector<int> index(shapeA.size());
                #pragma omp for
                for (size_t idx = 0; idx < out->grad.size(); ++idx) {
                    size_t tmp = idx;
                    for (int d = int(shapeA.size()) - 1; d >= 0; --d) {
                        index[d] = tmp % shapeA[d];
                        tmp /= shapeA[d];
                    }

                    size_t b_idx = 0;
                    for (int d = 0; d < shapeB.size(); ++d) {
                        int ix = index[d + offset];
                        if (shapeB[d] == 1) ix = 0;
                        b_idx += ix * b_strides[d];
                    }

                    #pragma omp atomic
                    B->grad[b_idx] -= out->grad[idx];
                }
            }
        }
    };
    return out;
}

// matmul
// (m, n) x (n, k) = (m, k)
// (B, m, n) x (n, k) = (B, m, k) 
// (B, m, n) x (B, n, k) = (B, m, k)
std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor>& A,
                               const std::shared_ptr<Tensor>& B) {
    bool requires_grad = A->requires_grad || B->requires_grad;

    if (A->shape.size() == 2 && B->shape.size() == 2) {
        int m = A->shape[0], n = A->shape[1], k = B->shape[1];
        assert(n == B->shape[0]);

        std::vector<float> out_data(m * k, 0.0f);
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < k; ++j) {
                float sum = 0.0f;
                for (int t = 0; t < n; ++t) {
                    sum += A->data[i*n + t] * B->data[t*k + j];
                }
                out_data[i*k + j] = sum;
            }
        }

        auto out = std::make_shared<Tensor>(out_data, std::vector<int>{m, k}, requires_grad);
        out->prev = {A, B};
        out->backward_fn = [A, B, out, m, n, k]() {
            if (A->requires_grad) {
                #pragma omp parallel for collapse(2)
                for (int i = 0; i < m; ++i) {
                    for (int t = 0; t < n; ++t) {
                        float sum = 0.0f;
                        for (int j = 0; j < k; ++j) {
                            sum += out->grad[i*k + j] * B->data[t*k + j];
                        }
                        A->grad[i*n + t] += sum;
                    }
                }
            }
            if (B->requires_grad) {
                #pragma omp parallel for collapse(2)
                for (int t = 0; t < n; ++t) {
                    for (int j = 0; j < k; ++j) {
                        float sum = 0.0f;
                        for (int i = 0; i < m; ++i) {
                            sum += A->data[i*n + t] * out->grad[i*k + j];
                        }
                        B->grad[t*k + j] += sum;
                    }
                }
            }
        };
        return out;
    }

    if (A->shape.size() == 3 && B->shape.size() == 2) {
        int batch_size = A->shape[0], m = A->shape[1], n = A->shape[2], k = B->shape[1];
        assert(n == B->shape[0]);

        std::vector<float> out_data(batch_size * m * k, 0.0f);
        #pragma omp parallel for collapse(2)
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < k; ++j) {
                    float sum = 0.0f;
                    for (int t = 0; t < n; ++t) {
                        sum += A->data[b*m*n + i*n + t] * B->data[t*k + j];
                    }
                    out_data[b*m*k + i*k + j] = sum;
                }
            }
        }

        auto out = std::make_shared<Tensor>(out_data, std::vector<int>{batch_size, m, k}, requires_grad);
        out->prev = {A, B};
        out->backward_fn = [A, B, out, batch_size, m, n, k]() {
            if (A->requires_grad) {
                #pragma omp parallel for collapse(3)
                for (int b = 0; b < batch_size; ++b)
                    for (int i = 0; i < m; ++i)
                        for (int t = 0; t < n; ++t) {
                            float sum = 0.0f;
                            for (int j = 0; j < k; ++j) {
                                sum += out->grad[b*m*k + i*k + j] * B->data[t*k + j];
                            }
                            A->grad[b*m*n + i*n + t] += sum;
                        }
            }
            if (B->requires_grad) {
                #pragma omp parallel for collapse(2)
                for (int t = 0; t < n; ++t)
                    for (int j = 0; j < k; ++j) {
                        float sum = 0.0f;
                        for (int b = 0; b < batch_size; ++b)
                            for (int i = 0; i < m; ++i)
                                sum += A->data[b*m*n + i*n + t] * out->grad[b*m*k + i*k + j];
                        B->grad[t*k + j] += sum;
                    }
            }
        };
        return out;
    }

    if (A->shape.size() == 3 && B->shape.size() == 3) {
        int batch_size = A->shape[0];
        int m = A->shape[1];
        int n = A->shape[2];
        int n2 = B->shape[1];
        int k = B->shape[2];
        assert(batch_size == B->shape[0]);
        assert(n == n2);

        std::vector<float> out_data(batch_size * m * k, 0.0f);
        #pragma omp parallel for collapse(3)
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < k; ++j) {
                    float sum = 0.0f;
                    for (int t = 0; t < n; ++t) {
                        sum += A->data[b*m*n + i*n + t] * B->data[b*n*k + t*k + j];
                    }
                    out_data[b*m*k + i*k + j] = sum;
                }
            }
        }

        auto out = std::make_shared<Tensor>(out_data,
            std::vector<int>{batch_size, m, k}, requires_grad);
        out->prev = {A, B};
        out->backward_fn = [A, B, out, batch_size, m, n, k]() {
            if (A->requires_grad) {
                #pragma omp parallel for collapse(4)
                for (int b = 0; b < batch_size; ++b)
                for (int i = 0; i < m; ++i)
                for (int t = 0; t < n; ++t) {
                    float sum = 0.0f;
                    for (int j = 0; j < k; ++j) {
                        sum += out->grad[b*m*k + i*k + j] * B->data[b*n*k + t*k + j];
                    }
                    A->grad[b*m*n + i*n + t] += sum;
                }
            }
            if (B->requires_grad) {
                #pragma omp parallel for collapse(4)
                for (int b = 0; b < batch_size; ++b)
                for (int t = 0; t < n; ++t)
                for (int j = 0; j < k; ++j) {
                    float sum = 0.0f;
                    for (int i = 0; i < m; ++i) {
                        sum += A->data[b*m*n + i*n + t] * out->grad[b*m*k + i*k + j];
                    }
                    B->grad[b*n*k + t*k + j] += sum;
                }
            }
        };
        return out;
    }

    throw std::invalid_argument("Unsupported shapes for matmul");
}

std::shared_ptr<Tensor> relu(const std::shared_ptr<Tensor>& A) {
    std::vector<float> out_data(A->data.size());
    #pragma omp parallel for
    for (size_t i = 0; i < A->data.size(); ++i) {
        out_data[i] = std::max(0.0f, A->data[i]);
    }
    auto out = std::make_shared<Tensor>(out_data, A->shape, A->requires_grad);
    out->prev = {A};
    out->backward_fn = [A, out]() {
        if (A->requires_grad) {
            #pragma omp parallel for
            for (size_t i = 0; i < A->grad.size(); ++i) {
                A->grad[i] += (A->data[i] > 0.0f ? 1.0f : 0.0f) * out->grad[i];
            }
        }
    };
    return out;
}

std::shared_ptr<Tensor> sigmoid(const std::shared_ptr<Tensor>& A) {
    std::vector<float> out_data(A->data.size());
    #pragma omp parallel for
    for (size_t i = 0; i < A->data.size(); ++i) {
        out_data[i] = 1.0f / (1.0f + std::exp(-A->data[i]));
    }
    auto out = std::make_shared<Tensor>(out_data, A->shape, A->requires_grad);
    out->prev = {A};
    out->backward_fn = [A, out]() {
        if (A->requires_grad) {
            #pragma omp parallel for
            for (size_t i = 0; i < A->grad.size(); ++i) {
                float s = out->data[i];
                A->grad[i] += out->grad[i] * s * (1.0f - s);
            }
        }
    };
    return out;
}

std::shared_ptr<Tensor> mse_loss(const std::shared_ptr<Tensor>& preds,
                                 const std::shared_ptr<Tensor>& labels) {
    assert(preds->shape == labels->shape);
    size_t N = preds->data.size();
    float sum = 0.0f;
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < N; ++i) {
        float d = preds->data[i] - labels->data[i];
        sum += d * d;
    }
    std::vector<float> loss_data(1, sum / float(N));
    auto loss = std::make_shared<Tensor>(loss_data,
                                         std::vector<int>{},
                                         preds->requires_grad || labels->requires_grad);
    loss->prev = {preds, labels};
    loss->backward_fn = [preds, labels, loss, N]() {
        #pragma omp parallel for
        for (size_t i = 0; i < N; ++i) {
            float d = preds->data[i] - labels->data[i];
            float g = 2.0f * d / float(N) * loss->grad[0];
            #pragma omp atomic
            preds->grad[i] += g;
            #pragma omp atomic
            labels->grad[i] -= g;
        }
    };
    return loss;
}

std::shared_ptr<Tensor> embedding(const std::shared_ptr<Tensor>& indices, 
                                  const std::shared_ptr<Tensor>& embedding_matrix) {
    const std::vector<int>& index_shape = indices->shape;
    const std::vector<float>& idx_data = indices->data;
    const std::vector<float>& E = embedding_matrix->data;
    int vocab_size = embedding_matrix->shape[0];
    int embedding_dim = embedding_matrix->shape[1];

    int num_indices = 1;
    for (int dim : index_shape) num_indices *= dim;

    std::vector<int> out_shape = index_shape;
    out_shape.push_back(embedding_dim);
    std::vector<float> out_data(num_indices * embedding_dim);

    #pragma omp parallel for
    for (int i = 0; i < num_indices; ++i) {
        int idx = static_cast<int>(idx_data[i]);
        assert(idx >= 0 && idx < vocab_size);

        const float* src = &E[idx * embedding_dim];
        float* dest = &out_data[i * embedding_dim];

        int d = 0;
        for (; d <= embedding_dim - 4; d += 4) {
            float32x4_t vec = vld1q_f32(src + d);
            vst1q_f32(dest + d, vec);
        }
        for (; d < embedding_dim; ++d) {
            dest[d] = src[d];
        }
    }

    auto out = std::make_shared<Tensor>(out_data, out_shape, embedding_matrix->requires_grad);
    out->prev = {embedding_matrix};
    out->backward_fn = [indices, embedding_matrix, out, num_indices, embedding_dim]() {
        if (!embedding_matrix->requires_grad) return;

        auto& E_grad = embedding_matrix->grad;
        auto& out_grad = out->grad;

        const std::vector<float>& idx_data = indices->data;

        #pragma omp parallel for
        for (int i = 0; i < num_indices; ++i) {
            int idx = static_cast<int>(idx_data[i]);
            for (int d = 0; d < embedding_dim; ++d) {
                #pragma omp atomic
                E_grad[idx * embedding_dim + d] += out_grad[i * embedding_dim + d];
            }
        }
    };

    return out;
}

std::shared_ptr<Tensor> cross_entropy_loss(const std::shared_ptr<Tensor>& preds, 
                                           const std::shared_ptr<Tensor>& targets) {
    // Preds   -> [B, V] -> preds is a probability distribution over all tokens
    // Targets -> [B] -> targets is the correct index of the next token
    std::vector<int> target_shape = targets->shape;
    if (target_shape.size() > 0 && target_shape.back() == 1) {
        target_shape.pop_back();
    }

    int num_samples = 1;
    for (int d : target_shape) num_samples *= d;

    std::vector<int> pred_shape = preds->shape;
    assert(pred_shape.size() == target_shape.size()+1);
    for (int i = 0; i < target_shape.size(); ++i) {
        assert(pred_shape[i] == target_shape[i]);
    }

    int V = pred_shape.back();
    const std::vector<float>& pred_data = preds->data;
    const std::vector<float>& target_data = targets->data;

    float loss_sum = 0.0f;

    #pragma omp parallel for reduction(+:loss_sum)
    for (int i = 0; i < num_samples; ++i) {
        int target_idx = static_cast<int>(target_data[i]);
        float p = pred_data[i * V + target_idx];
        loss_sum += -std::log(std::max(p, 1e-12f));
    }

    float mean_loss = loss_sum / num_samples;
    auto out = std::make_shared<Tensor>(std::vector<float>{mean_loss}, std::vector<int>{}, preds->requires_grad);

    out->prev = {preds, targets};
    out->backward_fn = [preds, targets, num_samples, V]() {
        if (!preds->requires_grad) return;

        std::vector<float>& grad = preds->grad;
        const std::vector<float>& pred_data = preds->data;
        const std::vector<float>& target_data = targets->data;

        #pragma omp parallel for
        for (int i = 0; i < num_samples; ++i) {
            int label = static_cast<int>(target_data[i]);
            const float* in = &pred_data[i * V];
            float* g = &grad[i * V];

            int v = 0;
            for (; v <= V - 4; v += 4) {
                int idx = i * V + v;
                float32x4_t p_vec = vld1q_f32(in + v);
                float32x4_t one_hot = vdupq_n_f32(0.0f);

                if (label >= v && label < v + 4) {
                    float tmp[4] = {0, 0, 0, 0};
                    tmp[label - v] = -1.0f;
                    one_hot = vld1q_f32(tmp);
                }

                float32x4_t safe_p = vmaxq_f32(p_vec, vdupq_n_f32(1e-12f));
                float32x4_t grad_vec = vdivq_f32(one_hot, safe_p);
                grad_vec = vmulq_n_f32(grad_vec, 1.0f / num_samples);

                float32x4_t existing_grad = vld1q_f32(g + v);
                existing_grad = vaddq_f32(existing_grad, grad_vec);
                vst1q_f32(g + v, existing_grad);
            }

            for (; v < V; ++v) {
                int idx = i * V + v;
                float delta = (v == label ? -1.0f : 0.0f);
                grad[idx] += delta / std::max(pred_data[idx], 1e-12f) / num_samples;
            }
        }
    };

    return out;
}

std::shared_ptr<Tensor> softmax_last_dim(const std::shared_ptr<Tensor>& A) {
    const std::vector<float>& in_data = A->data;
    const std::vector<int>& shape = A->shape;
    int last_dim = shape.back();
    int n_slices = A->data.size() / last_dim;

    std::vector<float> out_data(A->data.size());

    #pragma omp parallel for
    for (int s = 0; s < n_slices; ++s) {
        int offset = s * last_dim;

        float max_val = -1e30f;
        for (int i = 0; i < last_dim; ++i) {
            max_val = std::max(max_val, in_data[offset + i]);
        }

        float sum = 0.0f;
        int i = 0;

        for (; i + 4 <= last_dim; i += 4) {
            float32x4_t v = vld1q_f32(&in_data[offset + i]);
            float32x4_t vmax = vdupq_n_f32(max_val);
            float32x4_t shifted = vsubq_f32(v, vmax);

            float tmp[4];
            vst1q_f32(tmp, shifted);
            for (int j = 0; j < 4; ++j) {
                tmp[j] = std::exp(tmp[j]);
            }
            vst1q_f32(&out_data[offset + i], vld1q_f32(tmp));
            sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
        }

        for (; i < last_dim; ++i) {
            float ex = std::exp(in_data[offset + i] - max_val);
            out_data[offset + i] = ex;
            sum += ex;
        }

        float32x4_t vsum = vdupq_n_f32(sum);
        i = 0;
        for (; i <= last_dim - 4; i += 4) {
            float32x4_t out = vld1q_f32(&out_data[offset + i]);
            float32x4_t norm = vdivq_f32(out, vsum);
            vst1q_f32(&out_data[offset + i], norm);
        }
        for (; i < last_dim; ++i) {
            out_data[offset + i] /= sum;
        }
    }

    auto out = std::make_shared<Tensor>(out_data, shape, A->requires_grad);
    out->prev = {A};

    out->backward_fn = [A, out, last_dim, n_slices]() {
        if (!A->requires_grad) return;

        const std::vector<float>& s = out->data;
        const std::vector<float>& grad_out = out->grad;
        std::vector<float>& grad_in = A->grad;

        #pragma omp parallel for
        for (int s_idx = 0; s_idx < n_slices; ++s_idx) {
            int offset = s_idx * last_dim;

            float dot = 0.0f;
            for (int i = 0; i < last_dim; ++i) {
                dot += grad_out[offset + i] * s[offset + i];
            }

            for (int i = 0; i < last_dim; ++i) {
                grad_in[offset + i] += s[offset + i] * (grad_out[offset + i] - dot);
            }
        }
    };

    return out;
}


std::shared_ptr<Tensor> transpose_last2(const std::shared_ptr<Tensor>& A) {
    const std::vector<int>& shape = A->shape;
    int rank = shape.size();
    assert (rank >= 2);

    std::vector<int> new_shape = shape;
    int tmp = new_shape[rank - 1];
    new_shape[rank - 1] = new_shape[rank - 2];
    new_shape[rank - 2] = tmp;

    std::vector<float> out_data(A->data.size());

    std::vector<int> strides(rank, 1);
    std::vector<int> new_strides(rank, 1);
    for (int i = rank - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
        new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
    }

    size_t total = A->data.size();

    #pragma omp parallel for
    for (size_t flat = 0; flat < total; ++flat) {
        int tmp = flat;
        int new_flat = 0;

        for (int i = 0; i < rank; ++i) {
            int dim_idx = tmp / strides[i];
            tmp %= strides[i];

            int mapped_i = i;
            if (i == rank - 1) mapped_i = rank - 2;
            else if (i == rank - 2) mapped_i = rank - 1;

            new_flat += dim_idx * new_strides[mapped_i];
        }

        out_data[new_flat] = A->data[flat];
    }

    return std::make_shared<Tensor>(out_data, new_shape, A->requires_grad);
}


std::shared_ptr<Tensor> scale(const std::shared_ptr<Tensor>& A, float factor) {
    std::vector<float> out_data(A->data.size());
    for (size_t i = 0; i < A->data.size(); ++i) {
        out_data[i] = A->data[i] * factor;
    }

    auto out = std::make_shared<Tensor>(out_data, A->shape, A->requires_grad);
    out->prev = {A};

    out->backward_fn = [A, factor, out]() {
        if (!A->requires_grad) return;
        for (size_t i = 0; i < A->grad.size(); ++i) {
            A->grad[i] += out->grad[i] * factor;
        }
    };

    return out;
}


std::shared_ptr<Tensor> concat_last_dim(std::vector<std::shared_ptr<Tensor>> data) {
    assert(!data.empty());

    const auto& base_shape = data[0]->shape;
    int rank = base_shape.size();
    int last_dim = base_shape.back();

    int concat_dim = 0;
    std::vector<int> common_shape = base_shape;
    common_shape.pop_back();

    for (const auto& t : data) {
        assert(t->shape.size() == rank);
        for (int i = 0; i < rank - 1; ++i)
            assert(t->shape[i] == common_shape[i]);
        concat_dim += t->shape.back();
    }

    std::vector<int> out_shape = common_shape;
    out_shape.push_back(concat_dim);

    int inner_stride = 1;
    for (int dim : common_shape) inner_stride *= dim;

    std::vector<float> out_data(inner_stride * concat_dim);
    int offset = 0;

    for (const auto& t : data) {
        int block_size = t->shape.back();
        int batch = t->data.size() / block_size;

        for (int i = 0; i < batch; ++i)
            for (int j = 0; j < block_size; ++j)
                out_data[i * concat_dim + offset + j] = t->data[i * block_size + j];

        offset += block_size;
    }

    return std::make_shared<Tensor>(out_data, out_shape, false);
}