#include "op.hpp"

namespace llaisys::ops {// F16 转换函数（与其他文件中相同）
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

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    // 设置设备上下文
    core::context().setDevice(in->deviceType(), in->deviceId());
    
    // 检查输入参数
    if (in->ndim() != 2) {
        throw std::runtime_error("linear: input must be 2D tensor");
    }
    if (weight->ndim() != 2) {
        throw std::runtime_error("linear: weight must be 2D tensor");
    }
    if (out->ndim() != 2) {
        throw std::runtime_error("linear: output must be 2D tensor");
    }
    
    size_t batch_size = in->shape()[0];
    size_t in_features = in->shape()[1];
    size_t out_features = weight->shape()[0];
    
    // 检查维度匹配
    if (weight->shape()[1] != in_features) {
        throw std::runtime_error("linear: input features and weight input features mismatch");
    }
    if (out->shape()[0] != batch_size || out->shape()[1] != out_features) {
        throw std::runtime_error("linear: output shape mismatch");
    }
    
    // 检查数据类型一致性
    if (in->dtype() != weight->dtype() || in->dtype() != out->dtype()) {
        throw std::runtime_error("linear: all tensors must have same dtype");
    }
    
    // 检查bias
    if (bias && bias->ndim() != 1) {
        throw std::runtime_error("linear: bias must be 1D tensor");
    }
    if (bias && bias->shape()[0] != out_features) {
        throw std::runtime_error("linear: bias size mismatch");
    }
    if (bias && bias->dtype() != in->dtype()) {
        throw std::runtime_error("linear: bias dtype mismatch");
    }
    
    // 检查张量连续性
    if (!in->isContiguous() || !weight->isContiguous() || !out->isContiguous()) {
        throw std::runtime_error("linear: all tensors must be contiguous");
    }
    
    // 根据数据类型进行计算
    switch (in->dtype()) {
    case LLAISYS_DTYPE_F32: {
        const float* in_data = reinterpret_cast<const float*>(in->data());
        const float* weight_data = reinterpret_cast<const float*>(weight->data());
        const float* bias_data = bias ? reinterpret_cast<const float*>(bias->data()) : nullptr;
        float* out_data = reinterpret_cast<float*>(out->data());
        
        // 计算 Y = X * W^T + b
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < out_features; ++j) {
                float sum = 0.0f;
                
                // 矩阵乘法：X[i,:] * W[j,:]
                for (size_t k = 0; k < in_features; ++k) {
                    sum += in_data[i * in_features + k] * weight_data[j * in_features + k];
                }
                
                // 加偏置
                if (bias_data) {
                    sum += bias_data[j];
                }
                
                out_data[i * out_features + j] = sum;
            }
        }
        break;
    }
    case LLAISYS_DTYPE_F64: {
        const double* in_data = reinterpret_cast<const double*>(in->data());
        const double* weight_data = reinterpret_cast<const double*>(weight->data());
        const double* bias_data = bias ? reinterpret_cast<const double*>(bias->data()) : nullptr;
        double* out_data = reinterpret_cast<double*>(out->data());
        
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < out_features; ++j) {
                double sum = 0.0;
                
                for (size_t k = 0; k < in_features; ++k) {
                    sum += in_data[i * in_features + k] * weight_data[j * in_features + k];
                }
                
                if (bias_data) {
                    sum += bias_data[j];
                }
                
                out_data[i * out_features + j] = sum;
            }
        }
        break;
    }
    case LLAISYS_DTYPE_F16: {
        const uint16_t* in_data = reinterpret_cast<const uint16_t*>(in->data());
        const uint16_t* weight_data = reinterpret_cast<const uint16_t*>(weight->data());
        const uint16_t* bias_data = bias ? reinterpret_cast<const uint16_t*>(bias->data()) : nullptr;
        uint16_t* out_data = reinterpret_cast<uint16_t*>(out->data());
        
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < out_features; ++j) {
                float sum = 0.0f;
                
                for (size_t k = 0; k < in_features; ++k) {
                    float in_val = f16_to_float(in_data[i * in_features + k]);
                    float weight_val = f16_to_float(weight_data[j * in_features + k]);
                    sum += in_val * weight_val;
                }
                
                if (bias_data) {
                    float bias_val = f16_to_float(bias_data[j]);
                    sum += bias_val;
                }
                
                out_data[i * out_features + j] = float_to_f16(sum);
            }
        }
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        const uint16_t* in_data = reinterpret_cast<const uint16_t*>(in->data());
        const uint16_t* weight_data = reinterpret_cast<const uint16_t*>(weight->data());
        const uint16_t* bias_data = bias ? reinterpret_cast<const uint16_t*>(bias->data()) : nullptr;
        uint16_t* out_data = reinterpret_cast<uint16_t*>(out->data());
        
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < out_features; ++j) {
                float sum = 0.0f;
                
                for (size_t k = 0; k < in_features; ++k) {
                    float in_val = bf16_to_float(in_data[i * in_features + k]);
                    float weight_val = bf16_to_float(weight_data[j * in_features + k]);
                    sum += in_val * weight_val;
                }
                
                if (bias_data) {
                    float bias_val = bf16_to_float(bias_data[j]);
                    sum += bias_val;
                }
                
                out_data[i * out_features + j] = float_to_bf16(sum);
            }
        }
        break;
    }
    default:
        throw std::runtime_error("linear: unsupported data type");
    }
}
} // namespace llaisys::ops
