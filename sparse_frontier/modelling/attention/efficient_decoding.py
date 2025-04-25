import torch
from typing import Optional, Tuple, List
from .abstract_attention import AbstractAttention
from .abstract_attention import AttentionUtils
from vllm.attention.backends.flash_attn import flash_attn_with_kvcache


def _update_last_page(
    page_reps: torch.Tensor,
    keys: torch.Tensor,  # [1, num_heads, head_dim]
    cache_seqlens: int,
    page_size: int,
):
    """Update representations of the page containing the current token.
    
    Args:
        page_reps: Page representations tensor
        keys: Key tensor for the current token
        cache_seqlens: Cache sequence length as an integer
        page_size: Size of each page in the KV cache
    """
    current_page_idx = (cache_seqlens - 1) // page_size
        
    page_reps[current_page_idx, 0] = torch.minimum(
        page_reps[current_page_idx, 0],
        keys.squeeze(0)
    )
    
    page_reps[current_page_idx, 1] = torch.maximum(
        page_reps[current_page_idx, 1],
        keys.squeeze(0)
    )


def _select_pages(
    page_reps: torch.Tensor,
    query: torch.Tensor,
    cache_seqlens: int,
    page_size: int,
    page_budget: int,
    offsets: torch.Tensor,
) -> Tuple[torch.Tensor, int]:
    """Select most relevant pages based on query-page similarity.
    
    Args:
        page_reps: Page representations tensor
        query: Query tensor for the current token
        cache_seqlens: Cache sequence length as an integer
        page_size: Size of each page in the KV cache
        page_budget: Maximum number of pages to select
        offsets: Offsets for each KV head
        
    Returns:
        Tuple of (selected page indices, new cache sequence length)
    """
    current_page_idx = (cache_seqlens - 1) // page_size
    assert current_page_idx > 0 and current_page_idx >= page_budget + 1
    
    # All models have GQA
    query_squeezed = query.squeeze(0)
    group_size = query_squeezed.size(0) // page_reps.size(2)
    page_reps = page_reps[:current_page_idx].repeat_interleave(group_size, dim=2)
        
    scores = torch.einsum(
        'hd,prhd->hprd',
        query_squeezed,
        page_reps
    )

    scores = scores.max(dim=2).values  # [num_pages, num_heads, head_dim]
    scores = scores.sum(dim=-1)
    
    _, indices = torch.topk(
        scores,
        k=page_budget + 1,
        dim=1,
        sorted=True,
    )

    indices[:, -1] = current_page_idx
    new_cache_seqlens = cache_seqlens - (current_page_idx - page_budget) * page_size
    new_cache_seqlens = torch.full((query.shape[1],), new_cache_seqlens, device=query.device, dtype=torch.int32)

    active_pages = indices.int() + offsets
    return active_pages, new_cache_seqlens


_select_pages = torch.compile(_select_pages)
_update_last_page = torch.jit.script(_update_last_page)


class QuestAttention(AbstractAttention):
    """Quest attention for efficient decoding with dynamic page selection.
    
    Quest maintains min and max representations for each page of KV cache and uses
    them to dynamically select the most relevant pages during decoding.
    """
    
    def __init__(
        self,
        token_budget: int,
        page_size: int,
        max_input_tokens: int,
        max_output_tokens: int,
        num_layers: int,
    ):
        """Initialize Quest attention.
        
        Args:
            token_budget: Maximum number of tokens to attend to
            page_size: Size of each page in the KV cache
            max_input_tokens: Maximum input token length (from config)
            max_output_tokens: Maximum output token length (from config)
            num_layers: Number of transformer layers (from model config)
        """
        super().__init__()
        self.token_budget = token_budget
        self.page_size = page_size
        self.page_budget = token_budget // page_size
        assert token_budget % page_size == 0, "Token budget must be divisible by page size"
        
        self.max_pages = ((max_input_tokens + max_output_tokens) + page_size - 1) // page_size
        self.num_layers = num_layers
        
        # Page representations per layer
        self.page_reps_per_layer: List[Optional[torch.Tensor]] = [None] * num_layers
        self.offsets = None
        
    def _init_page_reps(
        self,
        keys: torch.Tensor,
        layer_idx: int = 0,
    ):
        """Initialize page representations during prefilling for a specific layer."""
        _, num_heads, seq_len, head_dim = keys.shape
        keys = keys.squeeze(0).transpose(0, 1)  # [seq_len, num_heads, head_dim]
        
        num_pages = (seq_len + self.page_size - 1) // self.page_size
        
        if self.page_reps_per_layer[layer_idx] is None:
            self.page_reps_per_layer[layer_idx] = torch.zeros(
                self.max_pages, 2, num_heads, head_dim,
                device=keys.device, dtype=keys.dtype
            )

        self.page_reps_per_layer[layer_idx][:, 0] = float('inf')
        self.page_reps_per_layer[layer_idx][:, 1] = float('-inf')

        if seq_len % self.page_size != 0:
            complete_pages = seq_len // self.page_size
            complete_keys = keys[:complete_pages * self.page_size].view(complete_pages, self.page_size, num_heads, head_dim)
            complete_min = complete_keys.amin(dim=1)
            complete_max = complete_keys.amax(dim=1)
            self.page_reps_per_layer[layer_idx][:complete_pages] = torch.stack([complete_min, complete_max], dim=1)
            
            remainder_keys = keys[complete_pages * self.page_size:]
            
            self.page_reps_per_layer[layer_idx][complete_pages] = torch.stack([
                remainder_keys.amin(dim=0),
                remainder_keys.amax(dim=0),
            ], dim=0)
        else:
            keys = keys.view(num_pages, self.page_size, num_heads, head_dim)
            self.page_reps_per_layer[layer_idx][:num_pages] = torch.stack([
                keys.amin(dim=1),
                keys.amax(dim=1),
            ], dim=1)
        
    def __call__(
        self,
        queries: torch.Tensor,  # [batch_size, num_heads, seq_len, head_dim]
        keys: torch.Tensor,     # [batch_size, num_kv_heads, seq_len, head_dim]
        values: torch.Tensor,   # [batch_size, num_kv_heads, seq_len, head_dim]
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            queries: Query tensor of shape [batch_size, num_heads, seq_len, head_dim]
            keys: Key tensor of shape [batch_size, num_kv_heads, seq_len, head_dim]
            values: Value tensor of shape [batch_size, num_kv_heads, seq_len, head_dim]
            layer_idx: Index of the current transformer layer
            
        Returns:
            Attention output tensor of shape [batch_size, num_heads, seq_len, head_dim]
        """
        self._init_page_reps(keys, layer_idx)

        sparsity = 1.0 - self.token_budget / queries.shape[2]
        self.layer_sparsity_statistics.append(torch.tensor(sparsity, device=queries.device))

        return AttentionUtils.flash_attention(queries, keys, values)
        
    def decode(
        self,
        query: torch.Tensor,  # [1, num_heads, head_dim]
        keys: torch.Tensor,   # [1, num_kv_heads, head_dim]
        values: torch.Tensor, # [1, num_kv_heads, head_dim]
        k_cache: torch.Tensor,  # [num_kv_heads, num_blocks, block_size, head_dim]
        v_cache: torch.Tensor,  # [num_kv_heads, num_blocks, block_size, head_dim]
        cache_seqlens: torch.Tensor,  # [num_heads]
        output: torch.Tensor, # [1, num_heads, head_dim]
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Compute attention during decoding with dynamic page selection for a specific layer.
        
        Instead of retrieving the selected KV cache, we pass the entire KV cache
        and the selected page indices to flash_attn_with_kvcache.
        
        Args:
            query: Query tensor for a single token [1, num_heads, head_dim]
            keys: Key tensor for the current token [1, num_kv_heads, head_dim]
            values: Value tensor for the current token [1, num_kv_heads, head_dim]
            k_cache: Key cache tensor [num_kv_heads, num_blocks, block_size, head_dim]
            v_cache: Value cache tensor [num_kv_heads, num_blocks, block_size, head_dim]
            cache_seqlens: Tensor of sequence lengths per head [num_heads]
            output: Output tensor to store results [1, num_heads, head_dim]
            layer_idx: Index of the current transformer layer
            
        Returns:
            Attention output tensor of shape [1, num_heads, head_dim]
        """
        num_kv_heads, num_blocks, block_size, head_size = k_cache.shape
        _, num_q_heads, _ = query.shape
        cache_seqlens_int = cache_seqlens[0].item()  # Convert to integer for page selection

        if self.offsets is None:
            offsets = torch.arange(num_kv_heads, device=query.device, dtype=torch.int32)
            offsets = offsets.repeat_interleave(num_q_heads // num_kv_heads)
            offsets = offsets.unsqueeze(1) * num_blocks
            self.offsets = offsets

        _update_last_page(
            page_reps=self.page_reps_per_layer[layer_idx],
            keys=keys,
            cache_seqlens=cache_seqlens_int,
            page_size=self.page_size
        )
        
        active_pages, new_cache_seqlens = _select_pages(
            page_reps=self.page_reps_per_layer[layer_idx],
            query=query,
            cache_seqlens=cache_seqlens_int,
            page_size=self.page_size,
            page_budget=self.page_budget,
            offsets=self.offsets,
        )

        flash_attn_with_kvcache(
            q=query.squeeze(0).unsqueeze(1).unsqueeze(1),
            k_cache=k_cache.view(num_kv_heads * num_blocks, block_size, 1, head_size),
            v_cache=v_cache.view(num_kv_heads * num_blocks, block_size, 1, head_size),
            block_table=active_pages,
            cache_seqlens=new_cache_seqlens,
            causal=True,
            out=output.squeeze(0).unsqueeze(1).unsqueeze(1),
        )
