#include <Layer/LayerNorm.hpp>

#include <cmath>
#include <omp.h>

LayerNorm::LayerNorm(int dim) : dim(dim) {
    std::vector<float> ones(dim, 1.0f);
    std::vector<float> zeros(dim, 0.0f);
    gamma = tensor(ones, std::vector<int>{dim}, true);
    beta = tensor(zeros, std::vector<int>{dim}, true);
}

std::shared_ptr<Tensor> LayerNorm::forward(std::shared_ptr<Tensor> input) {
    const auto& shape = input->shape;
    int64_t total = static_cast<int64_t>(input->data.size());
    int64_t ld = dim;
    int64_t slices = total / ld;

    const auto& x = input->data;
    auto out_data = std::vector<float>(total);
    const auto& g = gamma->data;
    const auto& b = beta->data;

    std::vector<float> means(slices);
    std::vector<float> vars(slices);
    std::vector<float> x_hat(total);

    #pragma omp parallel for
    for (int64_t s = 0; s < slices; ++s) {
        int64_t off = s * ld;
        float sum = 0.0f;
        for (int64_t i = 0; i < ld; ++i) sum += x[off + i];
        float mean = sum / ld;
        means[s] = mean;
        float vsum = 0.0f;
        for (int64_t i = 0; i < ld; ++i) {
            float d = x[off + i] - mean;
            vsum += d * d;
        }
        vars[s] = vsum / ld;
    }

    #pragma omp parallel for
    for (int64_t s = 0; s < slices; ++s) {
        int64_t off = s * ld;
        float inv_std = 1.0f / std::sqrt(vars[s] + eps);
        for (int64_t i = 0; i < ld; ++i) {
            int64_t idx = off + i;
            float norm = (x[idx] - means[s]) * inv_std;
            x_hat[idx] = norm;
            out_data[idx] = norm * g[i] + b[i];
        }
    }

    auto out = std::make_shared<Tensor>(out_data, shape, input->requires_grad);
    out->prev = {input, gamma, beta};
    out->backward_fn = [this, input, out, means, vars, x_hat](void) {
        auto& dx = input->grad;
        auto& dgamma = this->gamma->grad;
        auto& dbeta  = this->beta->grad;
        const auto& dy = out->grad;

        int64_t last_dim = this->dim;
        int64_t n_slices = dy.size() / last_dim;

        #pragma omp parallel for
        for (int64_t i = 0; i < last_dim; ++i) {
            float g_sum = 0.0f;
            float b_sum = 0.0f;
            for (int64_t s = 0; s < n_slices; ++s) {
                int64_t off = s * last_dim;
                g_sum += dy[off + i] * x_hat[off + i];
                b_sum += dy[off + i];
            }

            #pragma omp atomic
            dgamma[i] += g_sum;

            #pragma omp atomic
            dbeta[i]  += b_sum;
        }

        #pragma omp parallel for
        for (int64_t s = 0; s < n_slices; ++s) {
            int64_t offset = s * last_dim;
            float inv_std = 1.0f / std::sqrt(vars[s] + this->eps);

            float sum_dy = 0.0f;
            float sum_dy_xhat = 0.0f;
            for (int64_t i = 0; i < last_dim; ++i) {
                float dyi = dy[offset + i];
                sum_dy += dyi;
                sum_dy_xhat += dyi * x_hat[offset + i];
            }

            for (int64_t i = 0; i < last_dim; ++i) {
                int64_t idx = offset + i;
                float dyi = dy[idx];
                float xi_hat = x_hat[idx];
                float gi = this->gamma->data[i];

                float dx_i = (1.0f / last_dim) * gi * inv_std
                        * (last_dim * dyi - sum_dy - xi_hat * sum_dy_xhat);
                dx[idx] += dx_i;
            }
        }
    };

    return out;
}

std::vector<std::shared_ptr<Tensor>> LayerNorm::parameters() {
    return { gamma, beta };
}

std::shared_ptr<LayerNorm> layer_norm(int dim) {
    return std::make_shared<LayerNorm>(dim);
}