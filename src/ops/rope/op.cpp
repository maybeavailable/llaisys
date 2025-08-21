#include "op.hpp"

namespace llaisys::ops {
    
    
static float f16_to_float(uint16_t h) {
    union { uint32_t i; float f; } u;
    
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h & 0x7c00);
    uint32_t mant = (h & 0x03ff);
    
    if (exp == 0) {
        // 零或次正规数
        if (mant == 0) {
            u.i = sign;  // 零
        } else {
            // 次正规数，转换为正规数
            exp = 0x1c000;
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                exp -= 0x0400;
            }
            mant &= 0x03ff;
            u.i = sign | (exp << 13) | (mant << 13);
        }
    } else if (exp == 0x7c00) {
        // 无穷大或NaN
        u.i = sign | 0x7f800000 | (mant << 13);
    } else {
        // 正规数
        u.i = sign | ((exp + 0x1c000) << 13) | (mant << 13);
    }
    
    return u.f;
}

static uint16_t float_to_f16(float f) {
    union { uint32_t i; float f; } u;
    u.f = f;
    
    uint32_t i = u.i;
    uint32_t sign = (i >> 16) & 0x8000;
    int32_t exp = ((i >> 23) & 0xff) - 127 + 15;
    uint32_t mant = i & 0x007fffff;
    
    if (exp <= 0) {
        if (exp < -10) {
            // 太小，返回零
            return (uint16_t)sign;
        }
        // 次正规数
        mant |= 0x00800000;
        uint32_t shift = 14 - exp;
        if (shift < 32) {
            mant >>= shift;
        } else {
            mant = 0;
        }
        return (uint16_t)(sign | (mant >> 13));
    } else if (exp >= 31) {
        // 无穷大
        return (uint16_t)(sign | 0x7c00);
    }
    
    // 正规数
    return (uint16_t)(sign | (exp << 10) | (mant >> 13));
}

// BF16 转换函数（这个比较简单，因为 BF16 就是截断的 F32）
static float bf16_to_float(uint16_t h) {
    union { uint32_t i; float f; } u;
    u.i = ((uint32_t)h) << 16;
    return u.f;
}

static uint16_t float_to_bf16(float f) {
    union { uint32_t i; float f; } u;
    u.f = f;
    // 简单的截断，可以考虑添加舍入
    return (uint16_t)(u.i >> 16);
}


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
                    float a = f16_to_float(in_data[base_idx + j]);
                    float b = f16_to_float(in_data[base_idx + j + half_d]);
                    
                    // 计算输出
                    float a_out = a * cos_phi - b * sin_phi;
                    float b_out = b * cos_phi + a * sin_phi;
                    
                    // 转回 F16
                    out_data[base_idx + j] = float_to_f16(a_out);
                    out_data[base_idx + j + half_d] = float_to_f16(b_out);
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
                    float a = bf16_to_float(in_data[base_idx + j]);
                    float b = bf16_to_float(in_data[base_idx + j + half_d]);
                    
                    // 计算输出
                    float a_out = a * cos_phi - b * sin_phi;
                    float b_out = b * cos_phi + a * sin_phi;
                    
                    // 转回 BF16
                    out_data[base_idx + j] = float_to_bf16(a_out);
                    out_data[base_idx + j + half_d] = float_to_bf16(b_out);
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
