#include "op.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    // 设置设备上下文
    core::context().setDevice(vals->deviceType(), vals->deviceId());
    
    // 检查输入参数
    if (vals->ndim() != 1) {
        throw std::runtime_error("argmax: vals must be 1D tensor");
    }
    if (max_idx->ndim() != 1 || max_idx->numel() != 1) {
        throw std::runtime_error("argmax: max_idx must be 1D tensor with single element");
    }
    if (max_val->ndim() != 1 || max_val->numel() != 1) {
        throw std::runtime_error("argmax: max_val must be 1D tensor with single element");
    }
    
    size_t n = vals->numel();
    if (n == 0) {
        throw std::runtime_error("argmax: vals tensor is empty");
    }
    
    // 如果在设备上，需要先拷贝到CPU进行计算
    tensor_t vals_cpu = vals;
    if (vals->deviceType() != LLAISYS_DEVICE_CPU) {
        vals_cpu = vals->to(LLAISYS_DEVICE_CPU, 0);
    }
    
    // 根据数据类型进行argmax计算
    size_t max_index = 0;
    
    switch (vals->dtype()) {
    case LLAISYS_DTYPE_F32: {
        const float* data = reinterpret_cast<const float*>(vals_cpu->data());
        float max_value = data[0];
        for (size_t i = 1; i < n; ++i) {
            if (data[i] > max_value) {
                max_value = data[i];
                max_index = i;
            }
        }
        // 写入结果
        reinterpret_cast<float*>(max_val->data())[0] = max_value;
        break;
    }
    case LLAISYS_DTYPE_F64: {
        const double* data = reinterpret_cast<const double*>(vals_cpu->data());
        double max_value = data[0];
        for (size_t i = 1; i < n; ++i) {
            if (data[i] > max_value) {
                max_value = data[i];
                max_index = i;
            }
        }
        reinterpret_cast<double*>(max_val->data())[0] = max_value;
        break;
    }
    case LLAISYS_DTYPE_I32: {
        const int32_t* data = reinterpret_cast<const int32_t*>(vals_cpu->data());
        int32_t max_value = data[0];
        for (size_t i = 1; i < n; ++i) {
            if (data[i] > max_value) {
                max_value = data[i];
                max_index = i;
            }
        }
        reinterpret_cast<int32_t*>(max_val->data())[0] = max_value;
        break;
    }
    case LLAISYS_DTYPE_I64: {
        const int64_t* data = reinterpret_cast<const int64_t*>(vals_cpu->data());
        int64_t max_value = data[0];
        for (size_t i = 1; i < n; ++i) {
            if (data[i] > max_value) {
                max_value = data[i];
                max_index = i;
            }
        }
        reinterpret_cast<int64_t*>(max_val->data())[0] = max_value;
        break;
    }
    default:
        throw std::runtime_error("argmax: unsupported data type");
    }
    
    // 写入索引结果（假设max_idx是int64类型）
    reinterpret_cast<int64_t*>(max_idx->data())[0] = static_cast<int64_t>(max_index);
}
} // namespace llaisys::ops
