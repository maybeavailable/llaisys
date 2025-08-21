#include "op.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    // 设置设备上下文
    core::context().setDevice(in->deviceType(), in->deviceId());
    
    // 检查输入参数
    if (in->ndim() != 3) {
        throw std::runtime_error("rope: input must be 3D tensor [seqlen, nhead, d]");
    }
    if (out->ndim() != 3) {
        throw std::runtime_error("rope: output must be 3D tensor [seqlen, nhead, d]");
    }
    if (pos_ids->ndim() != 1) {
        throw std::runtime_error("rope: pos_ids must be 1D tensor [seqlen]");
    }
    if (pos_ids->dtype() != LLAISYS_DTYPE_I64) {
        throw std::runtime_error("rope: pos_ids must be Int64 type");
    }
    
    size_t seqlen = in->shape()[0];
    size_t nhead = in->shape()[1];
    size_t d = in->shape()[2];
    
    // 检查维度匹配
    if (pos_ids->shape()[0] != seqlen) {
        throw std::runtime_error("rope: pos_ids length must match seqlen");
    }
    if (out->shape()[0] != seqlen || out->shape()[1] != nhead || out->shape()[2] != d) {
        throw std::runtime_error("rope: output shape mismatch");
    }
    if (d % 2 != 0) {
        throw std::runtime_error("rope: d must be even");
    }
    
    // 检查数据类型一致性
    if (in->dtype() != out->dtype()) {
        throw std::runtime_error("rope: input and output must have same dtype");
    }
    
    // 检查张量连续性
    if (!in->isContiguous() || !out->isContiguous() || !pos_ids->isContiguous()) {
        throw std::runtime_error("rope: all tensors must be contiguous");
    }
    
    const int64_t* pos_data = reinterpret_cast<const int64_t*>(pos_ids->data());
    size_t half_d = d / 2;
    
    // 根据数据类型进行计算
    switch (in->dtype()) {
    case LLAISYS_DTYPE_F32: {
        const float* in_data = reinterpret_cast<const float*>(in->data());
        float* out_data = reinterpret_cast<float*>(out->data());
        
        for (size_t i = 0; i < seqlen; ++i) {
            int64_t pos = pos_data[i];
            
            for (size_t h = 0; h < nhead; ++h) {
                for (size_t j = 0; j < half_d; ++j) {
                    // 计算角度 phi_{i,j} = pos * theta^{-2j/d}
                    float phi = pos * std::pow(theta, -2.0f * j / d);
                    float cos_phi = std::cos(phi);
                    float sin_phi = std::sin(phi);
                    
                    // 获取 a_{i,j} 和 b_{i,j}
                    size_t base_idx = i * nhead * d + h * d;
                    float a = in_data[base_idx + j];
                    float b = in_data[base_idx + j + half_d];
                    
                    // 计算输出
                    // a'_{i,j} = a_{i,j} * cos(phi) - b_{i,j} * sin(phi)
                    // b'_{i,j} = b_{i,j} * cos(phi) + a_{i,j} * sin(phi)
                    out_data[base_idx + j] = a * cos_phi - b * sin_phi;
                    out_data[base_idx + j + half_d] = b * cos_phi + a * sin_phi;
                }
            }
        }
        break;
    }
    case LLAISYS_DTYPE_F64: {
        const double* in_data = reinterpret_cast<const double*>(in->data());
        double* out_data = reinterpret_cast<double*>(out->data());
        
        for (size_t i = 0; i < seqlen; ++i) {
            int64_t pos = pos_data[i];
            
            for (size_t h = 0; h < nhead; ++h) {
                for (size_t j = 0; j < half_d; ++j) {
                    double phi = pos * std::pow(theta, -2.0 * j / d);
                    double cos_phi = std::cos(phi);
                    double sin_phi = std::sin(phi);
                    
                    size_t base_idx = i * nhead * d + h * d;
                    double a = in_data[base_idx + j];
                    double b = in_data[base_idx + j + half_d];
                    
                    out_data[base_idx + j] = a * cos_phi - b * sin_phi;
                    out_data[base_idx + j + half_d] = b * cos_phi + a * sin_phi;
                }
            }
        }
        break;
    }
    case LLAISYS_DTYPE_F16: {
        const uint16_t* in_data = reinterpret_cast<const uint16_t*>(in->data());
        uint16_t* out_data = reinterpret_cast<uint16_t*>(out->data());
        
        for (size_t i = 0; i < seqlen; ++i) {
            int64_t pos = pos_data[i];
            
            for (size_t h = 0; h < nhead; ++h) {
                for (size_t j = 0; j < half_d; ++j) {
                    float phi = pos * std::pow(theta, -2.0f * j / d);
                    float cos_phi = std::cos(phi);
                    float sin_phi = std::sin(phi);
                    
                    size_t base_idx = i * nhead * d + h * d;
                    
                    // 转换为 float 进行计算
                    float a = static_cast<float>(in_data[base_idx + j]) / 1000.0f;
                    float b = static_cast<float>(in_data[base_idx + j + half_d]) / 1000.0f;
                    
                    // 计算输出
                    float a_out = a * cos_phi - b * sin_phi;
                    float b_out = b * cos_phi + a * sin_phi;
                    
                    // 转回 F16
                    out_data[base_idx + j] = static_cast<uint16_t>(a_out * 1000.0f);
                    out_data[base_idx + j + half_d] = static_cast<uint16_t>(b_out * 1000.0f);
                }
            }
        }
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        const uint16_t* in_data = reinterpret_cast<const uint16_t*>(in->data());
        uint16_t* out_data = reinterpret_cast<uint16_t*>(out->data());
        
        for (size_t i = 0; i < seqlen; ++i) {
            int64_t pos = pos_data[i];
            
            for (size_t h = 0; h < nhead; ++h) {
                for (size_t j = 0; j < half_d; ++j) {
                    float phi = pos * std::pow(theta, -2.0f * j / d);
                    float cos_phi = std::cos(phi);
                    float sin_phi = std::sin(phi);
                    
                    size_t base_idx = i * nhead * d + h * d;
                    
                    // 转换为 float 进行计算
                    float a = static_cast<float>(in_data[base_idx + j]) / 1000.0f;
                    float b = static_cast<float>(in_data[base_idx + j + half_d]) / 1000.0f;
                    
                    // 计算输出
                    float a_out = a * cos_phi - b * sin_phi;
                    float b_out = b * cos_phi + a * sin_phi;
                    
                    // 转回 BF16
                    out_data[base_idx + j] = static_cast<uint16_t>(a_out * 1000.0f);
                    out_data[base_idx + j + half_d] = static_cast<uint16_t>(b_out * 1000.0f);
                }
            }
        }
        break;
    }
    default:
        throw std::runtime_error("rope: unsupported data type");
    }
}
} // namespace llaisys::ops
