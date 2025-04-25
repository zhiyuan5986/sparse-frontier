import torch

from abc import ABC
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from flash_attn import flash_attn_func
from vllm.attention.backends.flash_attn import flash_attn_with_kvcache


class AttentionUtils:
    @staticmethod
    def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute attention using Flash Attention

        flash_attn_func expects BLHD format, so we need to convert the input tensors to this format.
        
        Args:
            q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
            k: Key tensor of shape (batch_size, num_kv_heads, seq_len, head_dim) 
            v: Value tensor of shape (batch_size, num_kv_heads, seq_len, head_dim)
            
        Returns:
            Attention output tensor of shape (batch_size, num_heads, seq_len, head_dim)
        """
        out = flash_attn_func(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            causal=True,
        )
        return out.transpose(1, 2)
    
    @staticmethod
    def reshape_kv_cache(
        kv_cache: torch.Tensor,
        target_block_size: int,
        max_blocks: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve keys and values from cache for attention computation.
        
        Args:
            kv_cache: [2, num_blocks, block_size, num_kv_heads, head_size]
            target_block_size: Target block size for reshaping
            
        Returns:
            k_cache: [num_kv_heads, num_blocks, target_block_size, head_size]
            v_cache: [num_kv_heads, num_blocks, target_block_size, head_size]
        """
        num_blocks, block_size, num_kv_heads, head_size = kv_cache[0].shape
        final_num_blocks = min(max_blocks, (num_blocks * block_size) // target_block_size)
        left_original_num_blocks = (final_num_blocks * target_block_size) // block_size

        k_cache = kv_cache[0, :left_original_num_blocks, :, :].view(num_kv_heads, final_num_blocks, target_block_size, head_size)
        v_cache = kv_cache[1, :left_original_num_blocks, :, :].view(num_kv_heads, final_num_blocks, target_block_size, head_size)

        return k_cache, v_cache


class AbstractAttention(ABC):
    """Base class for attention implementations (both prefilling and KV compression)"""
    def __init__(self):
        self.sparsity_statistics = []
        self.layer_sparsity_statistics = []
        self.block_table = None
        self.cache_batch_idx = None

    def reset_sparsity_statistics(self):
        """Reset the accumulated sparsity statistics."""
        self.sparsity_statistics = []
        self.layer_sparsity_statistics = []

    def sync_and_calc_layer_stats(self):
        # Ensure we have layer sparsity statistics to process
        if not self.layer_sparsity_statistics:
            raise AssertionError("Layer sparsity statistics list is empty. Make sure statistics are collected before syncing.")

        layer_sparsity = torch.stack(self.layer_sparsity_statistics).mean(dim=0, keepdim=True)

        if get_tensor_model_parallel_world_size() > 1:
            layer_sparsity = tensor_model_parallel_all_gather(layer_sparsity)

        self.sparsity_statistics.append(layer_sparsity.mean().item())
        self.layer_sparsity_statistics = []

    def calculate_sparsity(self) -> float:
        return sum(self.sparsity_statistics) / len(self.sparsity_statistics)
    
    def __call__(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        """Compute attention with pattern-specific masking.
        
        Args:
            queries: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
            keys: Key tensor of shape (batch_size, num_kv_heads, seq_len, head_dim)
            values: Value tensor of shape (batch_size, num_kv_heads, seq_len, head_dim)
            layer_idx: Index of the current transformer layer
        Returns:
            Attention output tensor of shape (batch_size, num_heads, seq_len, head_dim)
        """
        return AttentionUtils.flash_attention(queries, keys, values)

    def decode(
        self,
        query: torch.Tensor,  # [1, num_heads, head_dim]
        keys: torch.Tensor,   # [1, num_kv_heads, head_dim]
        values: torch.Tensor, # [1, num_kv_heads, head_dim]
        k_cache: torch.Tensor, # [num_kv_heads, num_blocks, block_size, head_dim]
        v_cache: torch.Tensor, # [num_kv_heads, num_blocks, block_size, head_dim]
        cache_seqlens: torch.Tensor,  # [num_heads]
        output: torch.Tensor, # [1, num_heads, head_dim]
        layer_idx: int,
    ) -> torch.Tensor:
        """Compute attention during decoding phase using flash_attn_with_kvcache.
        
        Args:
            query: Query tensor for a single token [1, num_heads, head_dim]
            keys: Key tensor for the current token [1, num_kv_heads, head_dim]
            values: Value tensor for the current token [1, num_kv_heads, head_dim]
            k_cache: Key cache tensor [num_kv_heads, num_blocks, block_size, head_dim]
            v_cache: Value cache tensor [num_kv_heads, num_blocks, block_size, head_dim]
            cache_seqlens: Tensor of sequence lengths per head [num_heads]
            output: Output tensor to store results [1, num_heads, head_dim]
            layer_idx: Index of the current transformer layer
        """
        _, num_q_heads, _ = query.shape
        num_kv_heads, num_blocks, block_size, head_size = k_cache.shape

        if self.block_table is None:
            block_indices = torch.arange(num_blocks * num_kv_heads, device=query.device, dtype=torch.int32).reshape(num_kv_heads, num_blocks)
            block_indices = block_indices.repeat(1, num_q_heads // num_kv_heads)
            self.block_table = block_indices.reshape(num_q_heads, num_blocks)
        
        flash_attn_with_kvcache(
            q=query.squeeze(0).unsqueeze(1).unsqueeze(1),
            k_cache=k_cache.view(num_kv_heads * num_blocks, block_size, 1, head_size),
            v_cache=v_cache.view(num_kv_heads * num_blocks, block_size, 1, head_size),
            block_table=self.block_table,
            cache_seqlens=cache_seqlens,
            causal=True,
            out=output.squeeze(0).unsqueeze(1).unsqueeze(1),
        )

    def kv_compress(
        self, 
        queries: torch.Tensor,  # [num_tokens, num_heads, head_dim]
        keys: torch.Tensor,     # [num_tokens, num_kv_heads, head_size] 
        values: torch.Tensor,   # [num_tokens, num_kv_heads, head_size]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compress KV cache after prefilling (default: no compression)
        
        Returns:
            tuple: (compressed_keys, compressed_values, seq_lens) where:
                - compressed_keys: [num_kv_heads, max_seq_len, head_size]
                - compressed_values: [num_kv_heads, max_seq_len, head_size]
                - seq_lens: [num_kv_heads] tensor with actual sequence length per head
        """
        # Default implementation: no compression, all tokens kept
        seq_lens = torch.full((keys.size(1),), keys.size(0), device=keys.device, dtype=torch.long)
        # Transpose keys and values to match the expected output shape
        keys_t = keys.transpose(0, 1)  # [num_kv_heads, num_tokens, head_size]
        values_t = values.transpose(0, 1)  # [num_kv_heads, num_tokens, head_size]
        return keys_t, values_t, seq_lens
