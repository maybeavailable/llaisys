from typing import Sequence, Optional, Dict, Any
from ..libllaisys import get_qwen2_api, DataType, DeviceType
from ..libllaisys.qwen2 import LlaisysQwen2Meta, LlaisysQwen2Model_p
from ..libllaisys.tensor import llaisysTensor_t
from ..libllaisys import LIB_LLAISYS

from pathlib import Path
try:
    import safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    safetensors = None

import numpy as np
import ctypes
import json
import re


class Qwen2:
    """Qwen2 Transformer model implementation"""

    def __init__(self, model_path: str, device: DeviceType = DeviceType.CPU, device_ids: list = None):
        """
        Initialize Qwen2 model
        
        Args:
            model_path: Path to model directory containing config.json and safetensors files
            device: Device type (CPU or NVIDIA)
            device_ids: List of device IDs to use for multi-device setup
        """
        self.model_path = Path(model_path)
        self.device = device
        self.device_ids = device_ids or [0]
        self.model: Optional[LlaisysQwen2Model_p] = None
        self.api = get_qwen2_api()
        
        # Load model configuration
        self.config = self._load_config()
        
        # Create model meta from config
        self.meta = self._create_meta_from_config()
        
        # Create the model
        self.model = self.api.create_model(self.meta, int(device), self.device_ids)
        if not self.model:
            raise RuntimeError("Failed to create Qwen2 model")
        
        # Load weights from safetensors files
        self._load_weights()

    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration from config.json"""
        config_path = self.model_path / "config.json"
        if not config_path.exists():
            # Provide default configuration for testing
            return {
                "hidden_size": 2048,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "num_key_value_heads": 16,
                "intermediate_size": 5504,
                "max_position_embeddings": 32768,
                "vocab_size": 151936,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0,
                "eos_token_id": 151645,
                "torch_dtype": "bfloat16"
            }
        
        import json
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _create_meta_from_config(self) -> LlaisysQwen2Meta:
        """Create LlaisysQwen2Meta from config"""
        config = self.config
        
        # Map dtype string to enum
        dtype_map = {
            "float32": DataType.F32,
            "float16": DataType.F16,
            "bfloat16": DataType.BF16,
            "float64": DataType.F64
        }
        dtype_str = config.get("torch_dtype", "float32")
        dtype = dtype_map.get(dtype_str, DataType.F32)
        
        # Calculate head dimension
        hs = config["hidden_size"]
        nh = config["num_attention_heads"]
        dh = hs // nh
        
        meta = LlaisysQwen2Meta()
        meta.dtype = int(dtype)
        meta.nlayer = config["num_hidden_layers"]
        meta.hs = hs
        meta.nh = nh
        meta.nkvh = config.get("num_key_value_heads", nh)
        meta.dh = dh
        meta.di = config["intermediate_size"]
        meta.maxseq = config.get("max_position_embeddings", 2048)
        meta.voc = config["vocab_size"]
        meta.epsilon = float(config.get("rms_norm_eps", 1e-6))
        meta.theta = float(config.get("rope_theta", 10000.0))
        meta.end_token = config.get("eos_token_id", 2)
        
        return meta

    def _load_weights(self):
        """Load model weights from safetensors files"""
        if not HAS_SAFETENSORS:
            print("Warning: safetensors not available, skipping weight loading")
            return
            
        weights = self.api.get_weights(self.model)
        if not weights:
            raise RuntimeError("Failed to get model weights")
        
        # Load weights from safetensors files
        safetensor_files = list(self.model_path.glob("*.safetensors"))
        if not safetensor_files:
            print("Warning: No safetensor files found, model weights not loaded")
            return
            
        for file in sorted(safetensor_files):
            print(f"Loading weights from {file}")
            data = safetensors.safe_open(file, framework="numpy", device="cpu")
            
            for name in data.keys():
                tensor_data = data.get_tensor(name)
                self._load_weight_tensor(name, tensor_data, weights)

    def _load_weight_tensor(self, name: str, data: np.ndarray, weights):
        """Load a single weight tensor"""
        # Map parameter names to weight tensors
        # This is a simplified mapping - a real implementation would need
        # more sophisticated name mapping based on the actual model structure
        
        if "embed_tokens" in name or "token_embed" in name:
            self._load_tensor_data(weights.contents.in_embed, data)
        elif "lm_head" in name or "output" in name:
            self._load_tensor_data(weights.contents.out_embed, data)
        elif "norm.weight" in name and "model.norm" in name:
            self._load_tensor_data(weights.contents.out_norm_w, data)
        elif "input_layernorm.weight" in name:
            # Extract layer index and load to appropriate layer
            layer_idx = self._extract_layer_index(name)
            if layer_idx is not None and layer_idx < self.meta.nlayer:
                layer_tensor = weights.contents.attn_norm_w[layer_idx]
                self._load_tensor_data(layer_tensor, data)
        # Add more mappings for other weight types...

    def _extract_layer_index(self, name: str) -> Optional[int]:
        """Extract layer index from parameter name"""
        import re
        match = re.search(r'layers\.(\d+)\.', name)
        return int(match.group(1)) if match else None

    def _load_tensor_data(self, tensor: llaisysTensor_t, data: np.ndarray):
        """Load numpy data into a tensor"""
        if not tensor:
            return
        
        # Get tensor data pointer
        tensor_ptr = LIB_LLAISYS.tensorGetData(tensor)
        if not tensor_ptr:
            return
        
        # Copy data (this is a simplified approach)
        # In practice, you'd need to handle data type conversion and device transfer
        data_size = data.nbytes
        ctypes.memmove(tensor_ptr, data.ctypes.data, data_size)

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 50,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ) -> Sequence[int]:
        """
        Generate text tokens from input sequence
        
        Args:
            inputs: Input token sequence
            max_new_tokens: Maximum number of new tokens to generate
            top_k: Top-k sampling parameter (not implemented yet)
            top_p: Top-p sampling parameter (not implemented yet) 
            temperature: Temperature for sampling (not implemented yet)
            
        Returns:
            Generated token sequence
        """
        if not self.model:
            raise RuntimeError("Model not initialized")
        
        generated_tokens = list(inputs)
        
        for _ in range(max_new_tokens):
            # Run inference on current sequence
            next_token = self.api.infer(self.model, generated_tokens)
            
            if next_token < 0:
                break  # Error occurred
            
            generated_tokens.append(next_token)
            
            # Check for end token
            if next_token == self.meta.end_token:
                break
        
        return generated_tokens

    def __del__(self):
        """Cleanup model resources"""
        if self.model:
            self.api.destroy_model(self.model)
            self.model = None
