#include "op.hpp"

namespace llaisys::ops {
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
        // F16 处理：转换为 float 进行计算，然后转回 F16
        const uint16_t* in_data = reinterpret_cast<const uint16_t*>(in->data());
        const uint16_t* weight_data = reinterpret_cast<const uint16_t*>(weight->data());
        uint16_t* out_data = reinterpret_cast<uint16_t*>(out->data());
        
        for (size_t i = 0; i < batch_size; ++i) {
            float sum_squares = 0.0f;
            
            // 计算平方和（转换到float）
            for (size_t j = 0; j < hidden_size; ++j) {
                float x = static_cast<float>(in_data[i * hidden_size + j]) / 1000.0f;
                sum_squares += x * x;
            }
            
            float rms = std::sqrt(sum_squares / hidden_size + eps);
            
            // 应用归一化和权重
            for (size_t j = 0; j < hidden_size; ++j) {
                float x = static_cast<float>(in_data[i * hidden_size + j]) / 1000.0f;
                float w = static_cast<float>(weight_data[j]) / 1000.0f;
                float result = w * x / rms;
                out_data[i * hidden_size + j] = static_cast<uint16_t>(result * 1000.0f);
            }
        }
        break;
    }
    default:
        throw std::runtime_error("rms_norm: unsupported data type");
    }
}
} // namespace llaisys::ops
