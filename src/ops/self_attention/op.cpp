#include "op.hpp"

namespace llaisys::ops {
// F16 转换函数（与其他文件中相同）
static float f16_to_float(uint16_t h) {
    union { uint32_t i; float f; } u;
    u.i = ((h & 0x8000) << 16) | (((h & 0x7c00) + 0x1c000) << 13) | ((h & 0x03ff) << 13);
    return u.f;
}

static uint16_t float_to_f16(float f) {
    union { uint32_t i; float f; } u;
    u.f = f;
    uint32_t i = u.i;
    uint16_t h = ((i >> 16) & 0x8000) | ((((i & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) | ((i >> 13) & 0x03ff);
    return h;
}

// BF16 转换函数
static float bf16_to_float(uint16_t h) {
    union { uint32_t i; float f; } u;
    u.i = ((uint32_t)h) << 16;
    return u.f;
}

static uint16_t float_to_bf16(float f) {
    union { uint32_t i; float f; } u;
    u.f = f;
    return (uint16_t)(u.i >> 16);
}

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    // 设置设备上下文
    core::context().setDevice(q->deviceType(), q->deviceId());
    
    // 检查输入参数
    if (q->ndim() != 3) {
        throw std::runtime_error("self_attention: q must be 3D tensor [seqlen, nhead, d]");
    }
    if (k->ndim() != 3) {
        throw std::runtime_error("self_attention: k must be 3D tensor [total_len, nkvhead, d]");
    }
    if (v->ndim() != 3) {
        throw std::runtime_error("self_attention: v must be 3D tensor [total_len, nkvhead, dv]");
    }
    if (attn_val->ndim() != 3) {
        throw std::runtime_error("self_attention: attn_val must be 3D tensor [seqlen, nhead, dv]");
    }
    
    size_t seqlen = q->shape()[0];
    size_t nhead = q->shape()[1];
    size_t d = q->shape()[2];
    size_t total_len = k->shape()[0];
    size_t nkvhead = k->shape()[1];
    size_t dv = v->shape()[2];
    
    // 检查维度匹配
    if (k->shape()[2] != d) {
        throw std::runtime_error("self_attention: k and q must have same d dimension");
    }
    if (v->shape()[0] != total_len || v->shape()[1] != nkvhead) {
        throw std::runtime_error("self_attention: v and k must have same total_len and nkvhead");
    }
    if (attn_val->shape()[0] != seqlen || attn_val->shape()[1] != nhead || attn_val->shape()[2] != dv) {
        throw std::runtime_error("self_attention: attn_val shape mismatch");
    }
    
    // 检查数据类型一致性
    if (q->dtype() != k->dtype() || q->dtype() != v->dtype() || q->dtype() != attn_val->dtype()) {
        throw std::runtime_error("self_attention: all tensors must have same dtype");
    }
    
    // 检查张量连续性
    if (!q->isContiguous() || !k->isContiguous() || !v->isContiguous() || !attn_val->isContiguous()) {
        throw std::runtime_error("self_attention: all tensors must be contiguous");
    }
    
    // 计算每个查询头对应的键值头（支持 Multi-Query 和 Grouped-Query Attention）
    size_t heads_per_group = nhead / nkvhead;
    
    // 根据数据类型进行计算
    switch (q->dtype()) {
    case LLAISYS_DTYPE_F32: {
        const float* q_data = reinterpret_cast<const float*>(q->data());
        const float* k_data = reinterpret_cast<const float*>(k->data());
        const float* v_data = reinterpret_cast<const float*>(v->data());
        float* out_data = reinterpret_cast<float*>(attn_val->data());
        
        for (size_t i = 0; i < seqlen; ++i) {
            for (size_t h = 0; h < nhead; ++h) {
                size_t kv_head = h / heads_per_group;
                
                // 计算注意力分数 A = Q * K^T * scale
                std::vector<float> scores(total_len);
                for (size_t j = 0; j < total_len; ++j) {
                    float score = 0.0f;
                    for (size_t dim = 0; dim < d; ++dim) {
                        float q_val = q_data[i * nhead * d + h * d + dim];
                        float k_val = k_data[j * nkvhead * d + kv_head * d + dim];
                        score += q_val * k_val;
                    }
                    scores[j] = score * scale;
                }
                
                // 应用因果掩码和softmax
                // 因果掩码：只能关注到当前位置及之前的位置
                size_t valid_len = std::min(total_len, i + 1);
                
                // 找到有效范围内的最大值（数值稳定性）
                float max_score = scores[0];
                for (size_t j = 1; j < valid_len; ++j) {
                    max_score = std::max(max_score, scores[j]);
                }
                
                // 计算softmax
                std::vector<float> attn_weights(total_len, 0.0f);
                float sum_exp = 0.0f;
                for (size_t j = 0; j < valid_len; ++j) {
                    attn_weights[j] = std::exp(scores[j] - max_score);
                    sum_exp += attn_weights[j];
                }
                
                // 归一化
                for (size_t j = 0; j < valid_len; ++j) {
                    attn_weights[j] /= sum_exp;
                }
                
                // 计算输出 Y = softmax(A) * V
                for (size_t dim = 0; dim < dv; ++dim) {
                    float output = 0.0f;
                    for (size_t j = 0; j < valid_len; ++j) {
                        float v_val = v_data[j * nkvhead * dv + kv_head * dv + dim];
                        output += attn_weights[j] * v_val;
                    }
                    out_data[i * nhead * dv + h * dv + dim] = output;
                }
            }
        }
        break;
    }
    case LLAISYS_DTYPE_F64: {
        const double* q_data = reinterpret_cast<const double*>(q->data());
        const double* k_data = reinterpret_cast<const double*>(k->data());
        const double* v_data = reinterpret_cast<const double*>(v->data());
        double* out_data = reinterpret_cast<double*>(attn_val->data());
        
        for (size_t i = 0; i < seqlen; ++i) {
            for (size_t h = 0; h < nhead; ++h) {
                size_t kv_head = h / heads_per_group;
                
                std::vector<double> scores(total_len);
                for (size_t j = 0; j < total_len; ++j) {
                    double score = 0.0;
                    for (size_t dim = 0; dim < d; ++dim) {
                        double q_val = q_data[i * nhead * d + h * d + dim];
                        double k_val = k_data[j * nkvhead * d + kv_head * d + dim];
                        score += q_val * k_val;
                    }
                    scores[j] = score * scale;
                }
                
                size_t valid_len = std::min(total_len, i + 1);
                
                double max_score = scores[0];
                for (size_t j = 1; j < valid_len; ++j) {
                    max_score = std::max(max_score, scores[j]);
                }
                
                std::vector<double> attn_weights(total_len, 0.0);
                double sum_exp = 0.0;
                for (size_t j = 0; j < valid_len; ++j) {
                    attn_weights[j] = std::exp(scores[j] - max_score);
                    sum_exp += attn_weights[j];
                }
                
                for (size_t j = 0; j < valid_len; ++j) {
                    attn_weights[j] /= sum_exp;
                }
                
                for (size_t dim = 0; dim < dv; ++dim) {
                    double output = 0.0;
                    for (size_t j = 0; j < valid_len; ++j) {
                        double v_val = v_data[j * nkvhead * dv + kv_head * dv + dim];
                        output += attn_weights[j] * v_val;
                    }
                    out_data[i * nhead * dv + h * dv + dim] = output;
                }
            }
        }
        break;
    }
    case LLAISYS_DTYPE_F16: {
        const uint16_t* q_data = reinterpret_cast<const uint16_t*>(q->data());
        const uint16_t* k_data = reinterpret_cast<const uint16_t*>(k->data());
        const uint16_t* v_data = reinterpret_cast<const uint16_t*>(v->data());
        uint16_t* out_data = reinterpret_cast<uint16_t*>(attn_val->data());
        
        for (size_t i = 0; i < seqlen; ++i) {
            for (size_t h = 0; h < nhead; ++h) {
                size_t kv_head = h / heads_per_group;
                
                std::vector<float> scores(total_len);
                for (size_t j = 0; j < total_len; ++j) {
                    float score = 0.0f;
                    for (size_t dim = 0; dim < d; ++dim) {
                        float q_val = f16_to_float(q_data[i * nhead * d + h * d + dim]);
                        float k_val = f16_to_float(k_data[j * nkvhead * d + kv_head * d + dim]);
                        score += q_val * k_val;
                    }
                    scores[j] = score * scale;
                }
                
                size_t valid_len = std::min(total_len, i + 1);
                
                float max_score = scores[0];
                for (size_t j = 1; j < valid_len; ++j) {
                    max_score = std::max(max_score, scores[j]);
                }
                
                std::vector<float> attn_weights(total_len, 0.0f);
                float sum_exp = 0.0f;
                for (size_t j = 0; j < valid_len; ++j) {
                    attn_weights[j] = std::exp(scores[j] - max_score);
                    sum_exp += attn_weights[j];
                }
                
                for (size_t j = 0; j < valid_len; ++j) {
                    attn_weights[j] /= sum_exp;
                }
                
                for (size_t dim = 0; dim < dv; ++dim) {
                    float output = 0.0f;
                    for (size_t j = 0; j < valid_len; ++j) {
                        float v_val = f16_to_float(v_data[j * nkvhead * dv + kv_head * dv + dim]);
                        output += attn_weights[j] * v_val;
                    }
                    out_data[i * nhead * dv + h * dv + dim] = float_to_f16(output);
                }
            }
        }
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        const uint16_t* q_data = reinterpret_cast<const uint16_t*>(q->data());
        const uint16_t* k_data = reinterpret_cast<const uint16_t*>(k->data());
        const uint16_t* v_data = reinterpret_cast<const uint16_t*>(v->data());
        uint16_t* out_data = reinterpret_cast<uint16_t*>(attn_val->data());
        
        for (size_t i = 0; i < seqlen; ++i) {
            for (size_t h = 0; h < nhead; ++h) {
                size_t kv_head = h / heads_per_group;
                
                std::vector<float> scores(total_len);
                for (size_t j = 0; j < total_len; ++j) {
                    float score = 0.0f;
                    for (size_t dim = 0; dim < d; ++dim) {
                        float q_val = bf16_to_float(q_data[i * nhead * d + h * d + dim]);
                        float k_val = bf16_to_float(k_data[j * nkvhead * d + kv_head * d + dim]);
                        score += q_val * k_val;
                    }
                    scores[j] = score * scale;
                }
                
                size_t valid_len = std::min(total_len, i + 1);
                
                float max_score = scores[0];
                for (size_t j = 1; j < valid_len; ++j) {
                    max_score = std::max(max_score, scores[j]);
                }
                
                std::vector<float> attn_weights(total_len, 0.0f);
                float sum_exp = 0.0f;
                for (size_t j = 0; j < valid_len; ++j) {
                    attn_weights[j] = std::exp(scores[j] - max_score);
                    sum_exp += attn_weights[j];
                }
                
                for (size_t j = 0; j < valid_len; ++j) {
                    attn_weights[j] /= sum_exp;
                }
                
                for (size_t dim = 0; dim < dv; ++dim) {
                    float output = 0.0f;
                    for (size_t j = 0; j < valid_len; ++j) {
                        float v_val = bf16_to_float(v_data[j * nkvhead * dv + kv_head * dv + dim]);
                        output += attn_weights[j] * v_val;
                    }
                    out_data[i * nhead * dv + h * dv + dim] = float_to_bf16(output);
                }
            }
        }
        break;
    }
    default:
        throw std::runtime_error("self_attention: unsupported data type");
    }
}

} // namespace llaisys::ops
