#include "op.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    // 设置设备上下文
    core::context().setDevice(weight->deviceType(), weight->deviceId());
    
    // 检查输入参数
    if (index->ndim() != 1) {
        throw std::runtime_error("embedding: index must be 1D tensor");
    }
    if (weight->ndim() != 2) {
        throw std::runtime_error("embedding: weight must be 2D tensor");
    }
    if (out->ndim() != 2) {
        throw std::runtime_error("embedding: out must be 2D tensor");
    }
    if (index->dtype() != LLAISYS_DTYPE_I64) {
        throw std::runtime_error("embedding: index must be Int64 type");
    }
    
    size_t seq_len = index->numel();
    size_t vocab_size = weight->shape()[0];
    size_t embed_dim = weight->shape()[1];
    
    // 检查输出张量形状
    if (out->shape()[0] != seq_len || out->shape()[1] != embed_dim) {
        throw std::runtime_error("embedding: output shape mismatch");
    }
    
    // 检查数据类型一致性
    if (out->dtype() != weight->dtype()) {
        throw std::runtime_error("embedding: output and weight must have same dtype");
    }
    
    // 获取索引数据
    const int64_t* idx_data = reinterpret_cast<const int64_t*>(index->data());
    
    // 根据数据类型进行embedding查找
    size_t element_size = weight->elementSize();
    
    for (size_t i = 0; i < seq_len; ++i) {
        int64_t idx = idx_data[i];
        
        // 检查索引范围
        if (idx < 0 || idx >= static_cast<int64_t>(vocab_size)) {
            throw std::runtime_error("embedding: index out of range");
        }
        
        // 计算源地址和目标地址
        const std::byte* src = weight->data() + idx * weight->strides()[0] * element_size;
        std::byte* dst = out->data() + i * out->strides()[0] * element_size;
        
        // 复制一行embedding
        if (weight->isContiguous() && out->isContiguous()) {
            // 如果都是连续的，可以直接内存复制
            std::memcpy(dst, src, embed_dim * element_size);
        } else {
            // 如果不连续，需要按元素复制
            for (size_t j = 0; j < embed_dim; ++j) {
                const std::byte* src_elem = src + j * weight->strides()[1] * element_size;
                std::byte* dst_elem = dst + j * out->strides()[1] * element_size;
                std::memcpy(dst_elem, src_elem, element_size);
            }
        }
    }
}
} // namespace llaisys::ops
