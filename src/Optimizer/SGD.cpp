#include <Optimizer/SGD.hpp>

#include <algorithm>
#include <omp.h>
#include <arm_neon.h>

SGD::SGD(std::vector<std::shared_ptr<Tensor>> parameters, float lr) 
    : params(std::move(parameters)), lr(lr) {}

void SGD::step() {
    constexpr float clip_norm = 1.0f;
    float total_sum = 0.0f;

    #pragma omp parallel reduction(+:total_sum)
    {
        float thread_sum = 0.0f;
        float32x4_t acc = vdupq_n_f32(0.0f);

        #pragma omp for nowait
        for (int p_idx = 0; p_idx < (int)params.size(); ++p_idx) {
            const auto& grad = params[p_idx]->grad;
            size_t i = 0, n = grad.size();

            for (; i + 4 <= n; i += 4) {
                float32x4_t g = vld1q_f32(&grad[i]);
                acc = vmlaq_f32(acc, g, g);
            }

            float32x2_t sum2 = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
            thread_sum += vget_lane_f32(sum2, 0) + vget_lane_f32(sum2, 1);

            for (; i < n; ++i) {
                thread_sum += grad[i] * grad[i];
            }
        }

        total_sum += thread_sum;
    }

    float scale = 1.0f;
    if (total_sum > clip_norm * clip_norm)
        scale = clip_norm / std::sqrt(total_sum);

    #pragma omp parallel for
    for (int p_idx = 0; p_idx < (int)params.size(); ++p_idx) {
        auto& param = params[p_idx];
        float32x4_t vscale = vdupq_n_f32(lr * scale);
        size_t i = 0, n = param->data.size();

        for (; i + 4 <= n; i += 4) {
            float32x4_t p = vld1q_f32(&param->data[i]);
            float32x4_t g = vld1q_f32(&param->grad[i]);
            float32x4_t step = vmulq_f32(g, vscale);
            vst1q_f32(&param->data[i], vsubq_f32(p, step));
        }

        for (; i < n; ++i) {
            param->data[i] -= lr * scale * param->grad[i];
        }
    }
}

void SGD::zero_grad() {
    #pragma omp parallel for
    for (int p_idx = 0; p_idx < (int)params.size(); ++p_idx) {
        auto& grad = params[p_idx]->grad;
        size_t i = 0, n = grad.size();
        float32x4_t zero = vdupq_n_f32(0.0f);

        for (; i + 4 <= n; i += 4) {
            vst1q_f32(&grad[i], zero);
        }
        for (; i < n; ++i) {
            grad[i] = 0.0f;
        }
    }
}