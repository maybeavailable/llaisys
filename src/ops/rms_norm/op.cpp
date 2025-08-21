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
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    // 设置设备上下文
    core::context().setDevice(in->deviceType(), in->deviceId());
    
    // 检查输入参数
    if (in->ndim() != 2) {
        throw std::runtime_error("rms_norm: input must be 2D tensor");
    }
    if (weight->ndim() != 1) {
        throw std::runtime_error("rms_norm: weight must be 1D tensor");
    }
    if (out->ndim() != 2) {
        throw std::runtime_error("rms_norm: output must be 2D tensor");
    }
    
    size_t batch_size = in->shape()[0];
    size_t hidden_size = in->shape()[1];
    
    // 检查维度匹配
    if (weight->shape()[0] != hidden_size) {
        throw std::runtime_error("rms_norm: weight size must match input hidden dimension");
    }
    if (out->shape()[0] != batch_size || out->shape()[1] != hidden_size) {
        throw std::runtime_error("rms_norm: output shape mismatch");
    }
    
    // 检查数据类型一致性
    if (in->dtype() != weight->dtype() || in->dtype() != out->dtype()) {
        throw std::runtime_error("rms_norm: all tensors must have same dtype");
    }
    
    // 检查张量连续性
    if (!in->isContiguous() || !weight->isContiguous() || !out->isContiguous()) {
        throw std::runtime_error("rms_norm: all tensors must be contiguous");
    }
    
    // 根据数据类型进行计算
    switch (in->dtype()) {
    case LLAISYS_DTYPE_F32: {
        const float* in_data = reinterpret_cast<const float*>(in->data());
        const float* weight_data = reinterpret_cast<const float*>(weight->data());
        float* out_data = reinterpret_cast<float*>(out->data());
        
        // 对每一行进行RMS normalization
        for (size_t i = 0; i < batch_size; ++i) {
            // 计算平方和
            float sum_squares = 0.0f;
            for (size_t j = 0; j < hidden_size; ++j) {
                float x = in_data[i * hidden_size + j];
                sum_squares += x * x;
            }
            
            // 计算RMS
            float rms = std::sqrt(sum_squares / hidden_size + eps);
            
            // 应用归一化和权重
            for (size_t j = 0; j < hidden_size; ++j) {
                float x = in_data[i * hidden_size + j];
                float w = weight_data[j];
                out_data[i * hidden_size + j] = w * x / rms;
            }
        }
        break;
    }
    case LLAISYS_DTYPE_F64: {
        const double* in_data = reinterpret_cast<const double*>(in->data());
        const double* weight_data = reinterpret_cast<const double*>(weight->data());
        double* out_data = reinterpret_cast<double*>(out->data());
        
        for (size_t i = 0; i < batch_size; ++i) {
            double sum_squares = 0.0;
            for (size_t j = 0; j < hidden_size; ++j) {
                double x = in_data[i * hidden_size + j];
                sum_squares += x * x;
            }
            
            double rms = std::sqrt(sum_squares / hidden_size + eps);
            
            for (size_t j = 0; j < hidden_size; ++j) {
                double x = in_data[i * hidden_size + j];
                double w = weight_data[j];
                out_data[i * hidden_size + j] = w * x / rms;
            }
        }
        break;
    }
    case LLAISYS_DTYPE_F16: {
        const uint16_t* in_data = reinterpret_cast<const uint16_t*>(in->data());
        const uint16_t* weight_data = reinterpret_cast<const uint16_t*>(weight->data());
        uint16_t* out_data = reinterpret_cast<uint16_t*>(out->data());
        
        for (size_t i = 0; i < batch_size; ++i) {
            float sum_squares = 0.0f;
            
            for (size_t j = 0; j < hidden_size; ++j) {
                float x = f16_to_float(in_data[i * hidden_size + j]);
                sum_squares += x * x;
            }
            
            float rms = std::sqrt(sum_squares / hidden_size + eps);
            
            for (size_t j = 0; j < hidden_size; ++j) {
                float x = f16_to_float(in_data[i * hidden_size + j]);
                float w = f16_to_float(weight_data[j]);
                float result = w * x / rms;
                out_data[i * hidden_size + j] = float_to_f16(result);
            }
        }
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        const uint16_t* in_data = reinterpret_cast<const uint16_t*>(in->data());
        const uint16_t* weight_data = reinterpret_cast<const uint16_t*>(weight->data());
        uint16_t* out_data = reinterpret_cast<uint16_t*>(out->data());
        
        for (size_t i = 0; i < batch_size; ++i) {
            float sum_squares = 0.0f;
            
            for (size_t j = 0; j < hidden_size; ++j) {
                float x = bf16_to_float(in_data[i * hidden_size + j]);
                sum_squares += x * x;
            }
            
            float rms = std::sqrt(sum_squares / hidden_size + eps);
            
            for (size_t j = 0; j < hidden_size; ++j) {
                float x = bf16_to_float(in_data[i * hidden_size + j]);
                float w = bf16_to_float(weight_data[j]);
                float result = w * x / rms;
                out_data[i * hidden_size + j] = float_to_bf16(result);
            }
        }
        break;
    }
    default:
        throw std::runtime_error("rms_norm: unsupported data type");
    }
}
} // namespace llaisys::ops
