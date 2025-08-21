#include "op.hpp"

namespace llaisys::ops {
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
    default:
        throw std::runtime_error("linear: unsupported data type");
    }
}
} // namespace llaisys::ops
