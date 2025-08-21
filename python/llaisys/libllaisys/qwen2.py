import ctypes
from typing import Optional
from .llaisys_types import llaisysDataType_t, llaisysDeviceType_t
from .tensor import llaisysTensor_t


# Qwen2 Meta structure
class LlaisysQwen2Meta(ctypes.Structure):
    _fields_ = [
        ("dtype", llaisysDataType_t),
        ("nlayer", ctypes.c_size_t),
        ("hs", ctypes.c_size_t),
        ("nh", ctypes.c_size_t),
        ("nkvh", ctypes.c_size_t),
        ("dh", ctypes.c_size_t),
        ("di", ctypes.c_size_t),
        ("maxseq", ctypes.c_size_t),
        ("voc", ctypes.c_size_t),
        ("epsilon", ctypes.c_float),
        ("theta", ctypes.c_float),
        ("end_token", ctypes.c_int64),
    ]


# Qwen2 Weights structure
class LlaisysQwen2Weights(ctypes.Structure):
    _fields_ = [
        ("in_embed", llaisysTensor_t),
        ("out_embed", llaisysTensor_t),
        ("out_norm_w", llaisysTensor_t),
        ("attn_norm_w", ctypes.POINTER(llaisysTensor_t)),
        ("attn_q_w", ctypes.POINTER(llaisysTensor_t)),
        ("attn_q_b", ctypes.POINTER(llaisysTensor_t)),
        ("attn_k_w", ctypes.POINTER(llaisysTensor_t)),
        ("attn_k_b", ctypes.POINTER(llaisysTensor_t)),
        ("attn_v_w", ctypes.POINTER(llaisysTensor_t)),
        ("attn_v_b", ctypes.POINTER(llaisysTensor_t)),
        ("attn_o_w", ctypes.POINTER(llaisysTensor_t)),
        ("mlp_norm_w", ctypes.POINTER(llaisysTensor_t)),
        ("mlp_gate_w", ctypes.POINTER(llaisysTensor_t)),
        ("mlp_up_w", ctypes.POINTER(llaisysTensor_t)),
        ("mlp_down_w", ctypes.POINTER(llaisysTensor_t)),
    ]


# Opaque pointer type for LlaisysQwen2Model
class LlaisysQwen2Model(ctypes.Structure):
    pass


LlaisysQwen2Model_p = ctypes.POINTER(LlaisysQwen2Model)
LlaisysQwen2Weights_p = ctypes.POINTER(LlaisysQwen2Weights)


def load_qwen2(lib):
    """Load Qwen2 model functions from the shared library"""
    
    # llaisysQwen2ModelCreate
    lib.llaisysQwen2ModelCreate.argtypes = [
        ctypes.POINTER(LlaisysQwen2Meta),  # meta
        llaisysDeviceType_t,               # device
        ctypes.POINTER(ctypes.c_int),      # device_ids
        ctypes.c_int                       # ndevice
    ]
    lib.llaisysQwen2ModelCreate.restype = LlaisysQwen2Model_p
    
    # llaisysQwen2ModelDestroy
    lib.llaisysQwen2ModelDestroy.argtypes = [LlaisysQwen2Model_p]
    lib.llaisysQwen2ModelDestroy.restype = None
    
    # llaisysQwen2ModelWeights
    lib.llaisysQwen2ModelWeights.argtypes = [LlaisysQwen2Model_p]
    lib.llaisysQwen2ModelWeights.restype = LlaisysQwen2Weights_p
    
    # llaisysQwen2ModelInfer
    lib.llaisysQwen2ModelInfer.argtypes = [
        LlaisysQwen2Model_p,               # model
        ctypes.POINTER(ctypes.c_int64),    # token_ids
        ctypes.c_size_t                    # ntoken
    ]
    lib.llaisysQwen2ModelInfer.restype = ctypes.c_int64


class Qwen2ModelAPI:
    """Python wrapper for Qwen2 model C API"""
    
    def __init__(self, lib):
        self.lib = lib
        load_qwen2(lib)
    
    def create_model(self, meta: LlaisysQwen2Meta, device_type: int, device_ids: list) -> Optional[LlaisysQwen2Model_p]:
        """Create a new Qwen2 model"""
        device_ids_array = (ctypes.c_int * len(device_ids))(*device_ids)
        result = self.lib.llaisysQwen2ModelCreate(
            ctypes.byref(meta),
            device_type,
            device_ids_array,
            len(device_ids)
        )
        return result if result else None
    
    def destroy_model(self, model: LlaisysQwen2Model_p) -> None:
        """Destroy a Qwen2 model and free resources"""
        if model:
            self.lib.llaisysQwen2ModelDestroy(model)
    
    def get_weights(self, model: LlaisysQwen2Model_p) -> Optional[LlaisysQwen2Weights_p]:
        """Get model weights structure"""
        if not model:
            return None
        result = self.lib.llaisysQwen2ModelWeights(model)
        return result if result else None
    
    def infer(self, model: LlaisysQwen2Model_p, token_ids: list) -> int:
        """Run inference and get next token prediction"""
        if not model or not token_ids:
            return -1
        
        token_array = (ctypes.c_int64 * len(token_ids))(*token_ids)
        result = self.lib.llaisysQwen2ModelInfer(
            model,
            token_array,
            len(token_ids)
        )
        return int(result)


# Global API instance (will be set when library is loaded)
qwen2_api: Optional[Qwen2ModelAPI] = None


def get_qwen2_api() -> Qwen2ModelAPI:
    """Get the global Qwen2 API instance"""
    global qwen2_api
    if qwen2_api is None:
        raise RuntimeError("Qwen2 API not initialized. Make sure to call load_qwen2_api first.")
    return qwen2_api


def load_qwen2_api(lib) -> None:
    """Initialize the global Qwen2 API instance"""
    global qwen2_api
    qwen2_api = Qwen2ModelAPI(lib)
