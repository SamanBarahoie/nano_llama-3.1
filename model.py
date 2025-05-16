import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
from utils import MultiHeadAttention, FeedForward, RMSNorm

@dataclass
class ModelArgs:
    """Configuration for the Transformer model, compatible with LLaMA 3.1."""
    dim: int = 4096
    n_heads: int = 32
    n_kv_heads: Optional[int] = 8  # For grouped query attention
    intermediate_dim: int = 11008
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = 1.0
    norm_eps: float = 1e-5
    max_batch_size: int = 2
    max_seq_len: int = 32
    flash: bool = True
    vocab_size: int = 128256  # Typical for LLaMA 3.1
    n_layers: int = 32  # Typical for LLaMA 3.1 8B
    rope_theta: float = 10000.0
    use_scaled_rope: bool = False

    def __post_init__(self):
        """Validate ModelArgs parameters."""
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})")
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.rope_theta <= 0:
            raise ValueError(f"rope_theta must be positive, got {self.rope_theta}")

def apply_scaling(freqs: torch.Tensor) -> torch.Tensor:
    """Apply frequency scaling for rotary embeddings, optimized for LLaMA-like models.
    
    Args:
        freqs (torch.Tensor): Input frequencies for rotary embeddings.
    
    Returns:
        torch.Tensor: Scaled frequencies with same dtype and device.
    """
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # Original LLaMA 3 context length
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / freqs
    mask_high = wavelen < high_freq_wavelen
    mask_low = wavelen > low_freq_wavelen
    mask_mid = ~(mask_high | mask_low)

    scaled_freqs = freqs.clone()
    scaled_freqs[mask_low] /= scale_factor
    
    if mask_mid.any():
        smooth = (old_context_len / wavelen[mask_mid] - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        scaled_freqs[mask_mid] = (1 - smooth) * freqs[mask_mid] / scale_factor + smooth * freqs[mask_mid]
    
    return scaled_freqs

def precompute_freqs_cis(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    use_scaled: bool = False
) -> torch.Tensor:
    """Precompute frequency-cosine-sine pairs for rotary embeddings.
    
    Args:
        head_dim (int): Dimension per attention head.
        max_seq_len (int): Maximum sequence length.
        theta (float): Base frequency for RoPE. Defaults to 10000.0.
        use_scaled (bool): Whether to apply frequency scaling. Defaults to False.
    
    Returns:
        torch.Tensor: Precomputed freqs_cis with shape (max_seq_len, head_dim // 2, 2).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    
    if use_scaled:
        freqs = apply_scaling(freqs.to(t.device))
    
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # Complex64
    return torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)

class DecoderBlock(nn.Module):
    """A Transformer decoder block with self-attention and feed-forward sub-layers.
    
    Args:
        args (ModelArgs): Model configuration.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = MultiHeadAttention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            intermediate_dim=args.intermediate_dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass of the decoder block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int): Starting position for KV caching.
            freqs_cis (torch.Tensor): Precomputed RoPE frequencies.
            mask (Optional[torch.Tensor]): Attention mask, if any.
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim).
        """
        hidden_states = x + self.attention(
            self.attention_norm(x), start_pos, freqs_cis, mask
        )
        output = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))
        return output

class llama(nn.Module):
    """A Transformer model for autoregressive language modeling, compatible with LLaMA 3.1.
    
    Args:
        params (ModelArgs): Model configuration.
    """
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.embedding_layer = nn.Embedding(params.vocab_size, params.dim)
        self.decoder_layers = nn.ModuleList(
            DecoderBlock(params) for _ in range(params.n_layers)
        )
        self.final_norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output_projection = nn.Linear(params.dim, params.vocab_size, bias=False)
        
        self.freqs_cis = precompute_freqs_cis(
            head_dim=params.dim // params.n_heads,
            max_seq_len=params.max_seq_len * 2,
            theta=params.rope_theta,
            use_scaled=params.use_scaled_rope
        )
        
        # Share embedding and output weights (optional, LLaMA-like)
        self.output_projection.weight = self.embedding_layer.weight

    def _create_causal_mask(
        self,
        seq_len: int,
        start_pos: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        """Create a causal attention mask for autoregressive decoding.
        
        Args:
            seq_len (int): Sequence length.
            start_pos (int): Starting position for KV caching.
            device (torch.device): Device for the mask.
            dtype (torch.dtype): Dtype for compatibility with input.
        
        Returns:
            Optional[torch.Tensor]: Causal mask or None if seq_len <= 1.
        """
        if seq_len <= 1:
            return None
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        if start_pos > 0:
            mask = torch.hstack([
                torch.zeros((seq_len, start_pos), device=device, dtype=dtype),
                mask
            ])
        return mask.type(dtype)

    def forward(
        self,
        input_tokens: torch.Tensor,
        start_pos: int = -1,
        targets: Optional[torch.Tensor] = None,
        ignore_index: int = -100
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of the Transformer model.
        
        Args:
            input_tokens (torch.Tensor): Input token indices of shape (batch_size, seq_len).
            start_pos (int): Starting position for KV caching (-1 for full sequence).
            targets (Optional[torch.Tensor]): Target token indices for loss computation.
            ignore_index (int): Index to ignore in loss computation (default: -100).
        
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Logits and optional loss.
        """
        batch_size, seq_len = input_tokens.shape
        if seq_len > self.params.max_seq_len:
            raise ValueError(f"Sequence length ({seq_len}) exceeds max_seq_len ({self.params.max_seq_len})")
        
        # Embed input tokens
        hidden_states = self.embedding_layer(input_tokens)
        
        # Move freqs_cis to the same device
        freqs_cis = self.freqs_cis.to(hidden_states.device)
        freqs_cis = freqs_cis[start_pos : start_pos + seq_len] if start_pos >= 0 else freqs_cis[:seq_len]
        
        # Create causal mask
        mask = self._create_causal_mask(
            seq_len=seq_len,
            start_pos=start_pos,
            device=input_tokens.device,
            dtype=hidden_states.dtype
        )
        
        # Process through decoder layers
        for layer in self.decoder_layers:
            hidden_states = layer(hidden_states, start_pos, freqs_cis, mask)
        
        # Final normalization and output projection
        hidden_states = self.final_norm(hidden_states)
        logits = self.output_projection(hidden_states).float()
        
        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                input=logits.transpose(1, 2),
                target=targets,
                reduction="mean",
                ignore_index=ignore_index
            )
        
        return logits, loss