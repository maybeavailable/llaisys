#include "op.hpp"
#include"functional"
namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    // 设置设备上下文
    core::context().setDevice(in->deviceType(), in->deviceId());
    
    // 检查输入参数
    if (in->shape() != out->shape()) {
        throw std::runtime_error("rearrange: input and output must have same shape");
    }
    
    // 检查数据类型一致性
    if (in->dtype() != out->dtype()) {
        throw std::runtime_error("rearrange: input and output must have same dtype");
    }
    
    // 检查设备一致性
    if (in->deviceType() != out->deviceType() || in->deviceId() != out->deviceId()) {
        throw std::runtime_error("rearrange: input and output must be on same device");
    }
    
    size_t element_size = in->elementSize();
    const auto& shape = in->shape();
    
    // 如果两个张量都是连续的且步长相同，可以直接内存复制
    if (in->isContiguous() && out->isContiguous() && in->strides() == out->strides()) {
        size_t total_bytes = in->numel() * element_size;
        std::memcpy(out->data(), in->data(), total_bytes);
        return;
    }
    
    // 否则需要按元素复制，使用递归方式处理多维索引
    std::function<void(std::size_t, std::vector<size_t>&)> copy_recursive;
    copy_recursive = [&](size_t dim, std::vector<size_t>& indices) {
        if (dim == shape.size()) {
            // 到达最后一维，计算偏移并复制元素
            size_t in_offset = 0;
            size_t out_offset = 0;
            
            for (size_t i = 0; i < indices.size(); ++i) {
                in_offset += indices[i] * in->strides()[i] * element_size;
                out_offset += indices[i] * out->strides()[i] * element_size;
            }
            
            const std::byte* src = in->data() + in_offset;
            std::byte* dst = out->data() + out_offset;
            std::memcpy(dst, src, element_size);
            return;
        }
        
        // 遍历当前维度
        for (size_t i = 0; i < shape[dim]; ++i) {
            indices[dim] = i;
            copy_recursive(dim + 1, indices);
        }
    };
    
    std::vector<size_t> indices(shape.size());
    copy_recursive(0, indices);
}
} // namespace llaisys::ops
