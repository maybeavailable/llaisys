#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    // 检查张量是否为连续内存布局
    const auto& shape = this->shape();
    const auto& strides = this->strides();
    if (shape.empty()) return true;
    size_t expected_stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        if (strides[i] != static_cast<ptrdiff_t>(expected_stride)) {
            return false;
        }
        expected_stride *= shape[i];
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    // 检查order合法性
    size_t ndim = this->ndim();
    if (order.size() != ndim) {
        throw std::runtime_error("Tensor::permute: order size mismatch with ndim");
    }
    std::vector<bool> seen(ndim, false);
    for (size_t i : order) {
        if (i >= ndim || seen[i]) {
            throw std::runtime_error("Tensor::permute: invalid or duplicate dimension in order");
        }
        seen[i] = true;
    }

    // 生成新shape和新strides
    std::vector<size_t> new_shape(ndim);
    std::vector<ptrdiff_t> new_strides(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        new_shape[i] = this->shape()[order[i]];
        new_strides[i] = this->strides()[order[i]];
    }

    TensorMeta new_meta{_meta.dtype, new_shape, new_strides};
    return std::make_shared<Tensor>(new_meta, _storage, _offset);
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
        // 检查新形状的元素数量是否与原张量一致
    size_t new_numel = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    if (new_numel != this->numel()) {
        throw std::runtime_error("Tensor::view: new shape is not compatible with original tensor (numel mismatch)");
    }

    // 仅支持连续张量的view（不涉及数据传输）
    if (!this->isContiguous()) {
        throw std::runtime_error("Tensor::view: only contiguous tensors can be viewed without data copy");
    }

    // 计算新步长（C-style）
    std::vector<ptrdiff_t> new_strides(shape.size());
    size_t stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        new_strides[i] = stride;
        stride *= shape[i];
    }
    TensorMeta new_meta{_meta.dtype, shape, new_strides};
    return std::make_shared<Tensor>(new_meta, _storage, _offset);
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    // 检查参数合法性
    size_t ndim = this->ndim();
    if (dim >= ndim) {
        throw std::runtime_error("Tensor::slice: dim out of range");
    }
    if (start > end || end > this->shape()[dim]) {
        throw std::runtime_error("Tensor::slice: invalid start/end");
    }

    // 新shape
    std::vector<size_t> new_shape = this->shape();
    new_shape[dim] = end - start;

    // 步长不变
    const std::vector<ptrdiff_t>& strides = this->strides();
    std::vector<ptrdiff_t> new_strides = strides;

    // 修复：offset计算 - strides是以元素为单位，需要转换为字节
    size_t new_offset = _offset + start * strides[dim] * this->elementSize();

    TensorMeta new_meta{_meta.dtype, new_shape, new_strides};
    return std::make_shared<Tensor>(new_meta, _storage, new_offset);
}

void Tensor::load(const void *src_) {
    // 获取当前设备上下文
    core::context().setDevice(this->deviceType(), this->deviceId());
    auto api = core::context().runtime().api();
    size_t bytes = this->numel() * this->elementSize();
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        // 直接拷贝到CPU内存
        std::memcpy(reinterpret_cast<void*>(this->data()), src_, bytes);
    } else {
        // 主机到设备拷贝
        api->memcpy_sync(reinterpret_cast<void*>(this->data()), src_, bytes, LLAISYS_MEMCPY_H2D);
    }
    
}

tensor_t Tensor::contiguous() const {
    // 已经连续则直接返回自身副本
    if (this->isContiguous()) {
        return std::make_shared<Tensor>(_meta, _storage, _offset);
    }
    // 创建连续副本
    auto contig = Tensor::create(this->shape(), this->dtype(), this->deviceType(), this->deviceId());
    // 拷贝数据
    core::context().setDevice(this->deviceType(), this->deviceId());
    auto api = core::context().runtime().api();
    size_t bytes = this->numel() * this->elementSize();
    api->memcpy_sync(reinterpret_cast<void*>(contig->data()), reinterpret_cast<const void*>(this->data()), bytes, LLAISYS_MEMCPY_D2D);
    return contig;
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    // 如果本身是连续的，直接用view
    if (this->isContiguous()) {
        return this->view(shape);
    }
    // 否则先变为连续再view
    auto contig = this->contiguous();
    return contig->view(shape);
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    // 如果设备类型和id一致，直接返回副本
    if (this->deviceType() == device_type && this->deviceId() == device) {
        return std::make_shared<Tensor>(_meta, _storage, _offset);
    }
    // 创建新张量
    auto new_tensor = Tensor::create(this->shape(), this->dtype(), device_type, device);
    // 拷贝数据
    core::context().setDevice(device_type, device);
    auto api = core::context().runtime().api();
    size_t bytes = this->numel() * this->elementSize();
    api->memcpy_sync(reinterpret_cast<void*>(new_tensor->data()), reinterpret_cast<const void*>(this->data()), bytes,
        (this->deviceType() == LLAISYS_DEVICE_CPU && device_type != LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_H2D :
        (this->deviceType() != LLAISYS_DEVICE_CPU && device_type == LLAISYS_DEVICE_CPU) ? LLAISYS_MEMCPY_D2H :
        LLAISYS_MEMCPY_D2D);
    return new_tensor;
}

} // namespace llaisys
