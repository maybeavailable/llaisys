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


void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    // 设置设备上下文
    core::context().setDevice(gate->deviceType(), gate->deviceId());
    
    // 检查输入参数
    if (gate->ndim() != 2) {
        throw std::runtime_error("swiglu: gate must be 2D tensor");
    }
    if (up->ndim() != 2) {
        throw std::runtime_error("swiglu: up must be 2D tensor");
    }
    if (out->ndim() != 2) {
        throw std::runtime_error("swiglu: out must be 2D tensor");
    }
    
    // 检查形状匹配
    if (gate->shape()[0] != up->shape()[0] || gate->shape()[1] != up->shape()[1]) {
        throw std::runtime_error("swiglu: gate and up must have same shape");
    }
    if (out->shape()[0] != gate->shape()[0] || out->shape()[1] != gate->shape()[1]) {
        throw std::runtime_error("swiglu: out shape must match gate and up");
    }
    
    // 检查数据类型一致性
    if (gate->dtype() != up->dtype() || gate->dtype() != out->dtype()) {
        throw std::runtime_error("swiglu: all tensors must have same dtype");
    }
    
    // 检查张量连续性
    if (!gate->isContiguous() || !up->isContiguous() || !out->isContiguous()) {
        throw std::runtime_error("swiglu: all tensors must be contiguous");
    }
    
    size_t total_elements = gate->numel();
    
    // 根据数据类型进行计算
    switch (gate->dtype()) {
    case LLAISYS_DTYPE_F32: {
        const float* gate_data = reinterpret_cast<const float*>(gate->data());
        const float* up_data = reinterpret_cast<const float*>(up->data());
        float* out_data = reinterpret_cast<float*>(out->data());
        
        for (size_t i = 0; i < total_elements; ++i) {
            float gate_val = gate_data[i];
            float up_val = up_data[i];
            
            // 计算 SiLU(gate) = gate / (1 + exp(-gate))
            float silu = gate_val / (1.0f + std::exp(-gate_val));
            
            // 计算 SwiGLU: out = up * SiLU(gate)
            out_data[i] = up_val * silu;
        }
        break;
    }
    case LLAISYS_DTYPE_F64: {
        const double* gate_data = reinterpret_cast<const double*>(gate->data());
        const double* up_data = reinterpret_cast<const double*>(up->data());
        double* out_data = reinterpret_cast<double*>(out->data());
        
        for (size_t i = 0; i < total_elements; ++i) {
            double gate_val = gate_data[i];
            double up_val = up_data[i];
            
            double silu = gate_val / (1.0 + std::exp(-gate_val));
            out_data[i] = up_val * silu;
        }
        break;
    }
    case LLAISYS_DTYPE_F16: {
        const uint16_t* gate_data = reinterpret_cast<const uint16_t*>(gate->data());
        const uint16_t* up_data = reinterpret_cast<const uint16_t*>(up->data());
        uint16_t* out_data = reinterpret_cast<uint16_t*>(out->data());
        
        for (size_t i = 0; i < total_elements; ++i) {
            float gate_val = f16_to_float(gate_data[i]);
            float up_val = f16_to_float(up_data[i]);
            
            // 计算 SiLU(gate) = gate / (1 + exp(-gate))
            float silu = gate_val / (1.0f + std::exp(-gate_val));
            
            // 计算 SwiGLU: out = up * SiLU(gate)
            float result = up_val * silu;
            out_data[i] = float_to_f16(result);
        }
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        const uint16_t* gate_data = reinterpret_cast<const uint16_t*>(gate->data());
        const uint16_t* up_data = reinterpret_cast<const uint16_t*>(up->data());
        uint16_t* out_data = reinterpret_cast<uint16_t*>(out->data());
        
        for (size_t i = 0; i < total_elements; ++i) {
            float gate_val = bf16_to_float(gate_data[i]);
            float up_val = bf16_to_float(up_data[i]);
            
            // 计算 SiLU(gate) = gate / (1 + exp(-gate))
            float silu = gate_val / (1.0f + std::exp(-gate_val));
            
            // 计算 SwiGLU: out = up * SiLU(gate)
            float result = up_val * silu;
            out_data[i] = float_to_bf16(result);
        }
        break;
    }
    default:
        throw std::runtime_error("swiglu: unsupported data type");
    }
}
} // namespace llaisys::ops
