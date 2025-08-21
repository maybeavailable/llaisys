#include "op.hpp"

namespace llaisys::ops {
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
        // F16 处理：转换为 float 进行计算，然后转回 F16
        const uint16_t* gate_data = reinterpret_cast<const uint16_t*>(gate->data());
        const uint16_t* up_data = reinterpret_cast<const uint16_t*>(up->data());
        uint16_t* out_data = reinterpret_cast<uint16_t*>(out->data());
        
        for (size_t i = 0; i < total_elements; ++i) {
            float gate_val = static_cast<float>(gate_data[i]) / 1000.0f;
            float up_val = static_cast<float>(up_data[i]) / 1000.0f;
            
            // 计算 SiLU(gate) = gate / (1 + exp(-gate))
            float silu = gate_val / (1.0f + std::exp(-gate_val));
            
            // 计算 SwiGLU: out = up * SiLU(gate)
            float result = up_val * silu;
            out_data[i] = static_cast<uint16_t>(result * 1000.0f);
        }
        break;
    }
    default:
        throw std::runtime_error("swiglu: unsupported data type");
    }
}
} // namespace llaisys::ops
