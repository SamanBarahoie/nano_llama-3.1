import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm) for Transformer models.

    This module normalizes the input using the RMSNorm formula:
    x * (1 / sqrt(mean(x^2) + eps)) * weight, where weight is a learnable parameter.

    Args:
        dim (int): Dimension of the input features (e.g., 4096 for LLaMA 8B).
        eps (float, optional): Small value for numerical stability. Defaults to 1e-6.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) 

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight.to(x.dtype) 
    

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads for Grouped Query Attention."""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary positional embeddings to input tensor.
    
    Args:
        x: Input tensor of shape (bs, seqlen, n_heads, head_dim)
        freqs_cis: Frequency tensor for rotary embeddings
    
    Returns:
        Tensor with rotary embeddings applied
    """
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)

@dataclass
class AttentionConfig:
    """Configuration for Multi-Head Attention."""
    dim: int
    n_heads: int
    n_kv_heads: Optional[int] = None
    flash: bool = False
    max_batch_size: int = 32
    max_seq_len: int = 2048

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.dim % self.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")

class KVCache(nn.Module):
    """Manages key-value caching for attention."""
    def __init__(self, batch_size: int, seq_length: int, n_kv_heads: int, head_dim: int, dtype: torch.dtype, device: torch.device):
        super().__init__()
        cache_shape = (batch_size, seq_length, n_kv_heads, head_dim)
        self.register_buffer("cache_k", torch.zeros(cache_shape, dtype=dtype, device=device))
        self.register_buffer("cache_v", torch.zeros(cache_shape, dtype=dtype, device=device))

    def update(self, start_pos: int, xk: torch.Tensor, xv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache with new key and value tensors."""
        bsz, seqlen = xk.size(0), xk.size(1)
        self.cache_k[:bsz, start_pos:start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos:start_pos + seqlen] = xv
        return self.cache_k[:bsz, :start_pos + seqlen], self.cache_v[:bsz, :start_pos + seqlen]

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism with optional caching and flash attention."""
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.head_dim = config.dim // config.n_heads
        self.num_query_heads = config.n_heads
        self.num_kv_heads = config.n_kv_heads
        self.num_repeats = config.n_heads // config.n_kv_heads

        self._init_projection_layers()
        self.cache = None  # Will be initialized in forward pass if needed

    def _init_projection_layers(self):
        """Initialize linear layers for Q, K, V, and output projection."""
        self.query_linear = nn.Linear(self.config.dim, self.num_query_heads * self.head_dim, bias=False)
        self.key_linear = nn.Linear(self.config.dim, self.num_kv_heads * self.head_dim, bias=False)
        self.value_linear = nn.Linear(self.config.dim, self.num_kv_heads * self.head_dim, bias=False)
        self.output_projection = nn.Linear(self.num_query_heads * self.head_dim, self.config.dim, bias=False)

    def _init_cache(self, x: torch.Tensor):
        """Initialize KV cache based on input tensor properties."""
        if self.cache is None:
            self.cache = KVCache(
                batch_size=self.config.max_batch_size,
                seq_length=self.config.max_seq_len,
                n_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                dtype=x.dtype,
                device=x.device
            )

    def project_qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project input tensor into Query, Key, and Value."""
        bsz, seqlen, _ = x.shape
        q = self.query_linear(x).view(bsz, seqlen, self.num_query_heads, self.head_dim)
        k = self.key_linear(x).view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.value_linear(x).view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        return q, k, v

    def apply_rotary_embedding(self, q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary positional embeddings to Query and Key."""
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)
        return q, k

    def prepare_attention_inputs(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare Q, K, V for attention by transposing and repeating K, V."""
        k = repeat_kv(k, self.num_repeats)
        v = repeat_kv(v, self.num_repeats)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def compute_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute scaled dot-product attention with optional flash attention."""
        if self.config.flash:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        weights = F.softmax(scores.float(), dim=-1).type_as(q)
        return torch.matmul(weights, v)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Merge attention heads into a single output tensor."""
        bsz, _, seqlen, _ = x.shape
        return x.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for Multi-Head Attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            start_pos: Starting position for caching
            freqs_cis: Frequency tensor for rotary embeddings
            mask: Optional attention mask

        Returns:
            Output tensor of shape (batch_size, seq_len, dim)
        """
        # 1. Initialize cache if needed
        self._init_cache(x)

        # 2. Project input to Q, K, V
        q, k, v = self.project_qkv(x)

        # 3. Apply rotary embeddings
        q, k = self.apply_rotary_embedding(q, k, freqs_cis)

        # 4. Update cache if start_pos > 0 (indicating incremental decoding)
        if start_pos > 0:
            k, v = self.cache.update(start_pos, k, v)

        # 5. Prepare attention inputs
        q, k, v = self.prepare_attention_inputs(q, k, v)

        # 6. Compute attention
        attn_output = self.compute_attention(q, k, v, mask)

        # 7. Merge heads and project output
        merged = self.merge_heads(attn_output)
        return self.output_projection(merged)
    



class FeedForward(nn.Module):
    """Feed-Forward Network (FFN) with SwiGLU activation for Transformer models.
    
    This module implements a Feed-Forward Network with a SwiGLU activation function,
    commonly used in models like LLaMA. It projects the input to an intermediate
    dimension, applies SiLU activation and gating, and projects back to the input
    dimension.
    
    Args:
        dim (int): Input and output dimension of the FFN (e.g., 4096 for LLaMA 8B).
        intermediate_dim (int): Initial intermediate dimension before scaling.
        multiple_of (int): Ensures the intermediate dimension is a multiple of this value.
        ffn_dim_multiplier (Optional[float]): Optional scaling factor for intermediate dimension.
    """
    
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float] = None,
    ):
        super().__init__()
        
        # Validate input parameters
        self._validate_inputs(dim, intermediate_dim, multiple_of, ffn_dim_multiplier)
        
        # Compute the adjusted intermediate dimension
        self.intermediate_dim = self._compute_intermediate_dim(
            intermediate_dim, ffn_dim_multiplier, multiple_of
        )
        
        # Initialize linear layers
        self.up_proj = nn.Linear(dim, self.intermediate_dim, bias=False)  # Up projection
        self.gate_proj = nn.Linear(dim, self.intermediate_dim, bias=False)  # Gate projection
        self.down_proj = nn.Linear(self.intermediate_dim, dim, bias=False)  # Down projection

    def _validate_inputs(
        self,
        dim: int,
        intermediate_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ) -> None:
        """Validate input parameters to ensure they are positive and valid."""
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if intermediate_dim <= 0:
            raise ValueError(f"intermediate_dim must be positive, got {intermediate_dim}")
        if multiple_of <= 0:
            raise ValueError(f"multiple_of must be positive, got {multiple_of}")
        if ffn_dim_multiplier is not None and ffn_dim_multiplier <= 0:
            raise ValueError(f"ffn_dim_multiplier must be positive, got {ffn_dim_multiplier}")

    def _compute_intermediate_dim(
        self,
        intermediate_dim: int,
        ffn_dim_multiplier: Optional[float],
        multiple_of: int,
    ) -> int:
        """Compute the adjusted intermediate dimension for the FFN.
        
        Args:
            intermediate_dim (int): Initial intermediate dimension.
            ffn_dim_multiplier (Optional[float]): Scaling factor, if provided.
            multiple_of (int): Ensures the final dimension is a multiple of this value.
        
        Returns:
            int: Adjusted intermediate dimension.
        """
        # Reduce intermediate_dim to 2/3 of the initial value
        dim = int(2 * intermediate_dim / 3)
        
        # Apply scaling factor if provided
        if ffn_dim_multiplier is not None:
            dim = int(ffn_dim_multiplier * dim)
        
        # Round up to the nearest multiple of multiple_of
        dim = multiple_of * ((dim + multiple_of - 1) // multiple_of)
        
        return dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Feed-Forward Network with SwiGLU activation.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim).
        """
        # Apply up projection, SiLU activation, and gate projection
        up_output = F.silu(self.up_proj(x))
        gate_output = self.gate_proj(x)
        
        # Element-wise multiplication (SwiGLU)
        gated_output = up_output * gate_output
        
        # Apply down projection to return to input dimension
        return self.down_proj(gated_output)