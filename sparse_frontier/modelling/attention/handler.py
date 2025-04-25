import torch
from .abstract_attention import AttentionUtils


def update_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    is_prefilling: bool,
    tokens_per_head: torch.Tensor,
    q_heads_per_kv: int,
    head_indices: torch.Tensor = None,
    queries: torch.Tensor = None,
):
    """Update the KV cache with new key/value tensors.
    
    Args:
        key: [num_tokens, num_kv_heads, head_size]
        value: [num_tokens, num_kv_heads, head_size]
        k_cache: [num_kv_heads, num_blocks, block_size, head_size]
        v_cache: [num_kv_heads, num_blocks, block_size, head_size]
        is_prefilling: Whether we're in prefilling phase
        tokens_per_head: Tensor tracking token counts per head for current layer
        q_heads_per_kv: Number of query heads per key/value head
        head_indices: Precomputed indices for heads to avoid recreating tensor each time
        queries: [num_tokens, num_heads, head_size], required for compression
    """
    if is_prefilling:
        from sparse_frontier.modelling.attention.registry import get_attention
        key, value, seq_lens = get_attention().kv_compress(
            queries=queries,
            keys=key,
            values=value,
        )
        
        num_kv_heads, num_tokens, _ = key.shape
    
        # View the cache as a contiguous vector by merging block dimensions
        k_cache_flat = k_cache.view(num_kv_heads, -1, k_cache.shape[-1])
        v_cache_flat = v_cache.view(num_kv_heads, -1, v_cache.shape[-1])
        
        # Fill the prefix of the flattened cache
        k_cache_flat[:, :num_tokens] = key
        v_cache_flat[:, :num_tokens] = value
        
        tokens_per_head += seq_lens.repeat_interleave(q_heads_per_kv)
    else:
        num_tokens, num_kv_heads, _ = key.shape

        # Decoding case: single token update, use tokens_per_head to place correctly
        assert num_tokens == 1, "Decoding should only add one token at a time"
        
        # View the cache as a contiguous vector by merging block dimensions
        k_cache_flat = k_cache.view(num_kv_heads, -1, k_cache.shape[-1])
        v_cache_flat = v_cache.view(num_kv_heads, -1, v_cache.shape[-1])

        # Update the cache at the specific positions for all heads at once
        k_cache_flat[head_indices, tokens_per_head[::q_heads_per_kv]] = key[0]
        v_cache_flat[head_indices, tokens_per_head[::q_heads_per_kv]] = value[0]
        
        # Update token counts
        tokens_per_head += 1


class AttentionHandler:
    def __init__(
        self,
        tp_size: int,
        model_q_heads: int,
        model_kv_heads: int,
        model_layers: int,
        max_input_tokens: int,
        max_output_tokens: int,
        block_size: int
    ):
        """Initialize the attention handler.
        
        Args:
            tp_size: Tensor parallelism size
            model_q_heads: Total number of query heads
            model_kv_heads: Total number of key/value heads
            model_layers: Number of transformer layers
            max_input_tokens: Maximum number of input tokens
            max_output_tokens: Maximum number of output tokens
            block_size: default size of each block in the KV cache
        """
        self.tp_size = tp_size
        self.model_q_heads = model_q_heads
        self.model_kv_heads = model_kv_heads
        self.q_heads_per_gpu = model_q_heads // tp_size
        self.kv_heads_per_gpu = model_kv_heads // tp_size
        self.q_heads_per_kv = model_q_heads // model_kv_heads
        self.model_layers = model_layers
        self.max_seq_len = max_input_tokens + max_output_tokens
        self.max_blocks = (self.max_seq_len + block_size - 1) // block_size
        self.block_size = block_size
        
        self.current_layer = 0
        self.tokens_per_layer_head = torch.zeros(
            (model_layers, self.q_heads_per_gpu), 
            dtype=torch.int32, 
            device="cpu"
        )
        self.head_indices = None

    def _compute_attention_head_by_head(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention output by processing each head separately.
        
        Args:
            queries: [num_tokens, num_heads, head_size]
            keys: [num_tokens, num_kv_heads, head_size]
            values: [num_tokens, num_kv_heads, head_size]
            output: [num_tokens, num_heads, head_size]
        """
        from sparse_frontier.modelling.attention.registry import get_attention

        # Get the attention implementation
        attention = get_attention()
        
        efficient_attention_classes = [
            "BlockSparseAttentionMInference", 
            "VerticalAndSlashAttentionMInference", 
            "FlexPrefill"
        ]
        
        if attention.__class__.__name__ not in efficient_attention_classes:
            output[:] = attention(
                queries=queries.transpose(0, 1).unsqueeze(0),
                keys=keys.transpose(0, 1).unsqueeze(0),
                values=values.transpose(0, 1).unsqueeze(0),
                layer_idx=self.current_layer,
            ).squeeze(0).transpose(0, 1)
            return
        
        # Otherwise, process head by head
        head_size = queries.shape[-1]
        q_heads_per_kv = queries.shape[1] // keys.shape[1]

        for head in range(self.q_heads_per_gpu):
            kv_head = head // q_heads_per_kv
            output[:, head, :] = attention(
                queries=queries[:, head, :].view(1, 1, -1, head_size),
                keys=keys[:, kv_head, :].view(1, 1, -1, head_size),
                values=values[:, kv_head, :].view(1, 1, -1, head_size),
                layer_idx=self.current_layer,
            ).view(-1, head_size)
        
    def __call__(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Process attention for either prefill or decode phase.
        
        Args:
            queries: [num_tokens, num_heads, head_size]
            keys: [num_tokens, num_kv_heads, head_size]
            values: [num_tokens, num_kv_heads, head_size]
            kv_cache: [2, num_blocks, block_size, num_kv_heads, head_size]
            output: [num_tokens, num_heads, head_size]
        
        Returns:
            attention_output: [num_tokens, num_heads, head_size]
        """
        num_tokens = queries.shape[0]
        is_prefilling = num_tokens > 1
        
        if is_prefilling:
            self.tokens_per_layer_head[self.current_layer, :] = 0

        # Initialize head_indices if not already done
        if self.head_indices is None and keys.numel() > 0:
            num_kv_heads = keys.shape[1]
            self.head_indices = torch.arange(num_kv_heads, device=keys.device)
            self.tokens_per_layer_head = self.tokens_per_layer_head.to(keys.device)

        if kv_cache.numel() == 0:
            self._compute_attention_head_by_head(
                queries=queries,
                keys=keys,
                values=values,
                output=output,
            )
            self.current_layer = (self.current_layer + 1) % self.model_layers
            return
        
        if is_prefilling:
            self._compute_attention_head_by_head(
                queries=queries,
                keys=keys,
                values=values,
                output=output,
            )

        k_cache, v_cache = AttentionUtils.reshape_kv_cache(kv_cache, self.block_size, self.max_blocks)

        # Then update cache (with compression if prefilling)
        update_kv_cache(
            keys,
            values,
            k_cache,
            v_cache,
            is_prefilling=is_prefilling,
            tokens_per_head=self.tokens_per_layer_head[self.current_layer],
            q_heads_per_kv=self.q_heads_per_kv,
            head_indices=self.head_indices,
            queries=queries if is_prefilling else None,
        )

        if is_prefilling:
            # By this point we should have put all the layer sparsity statistics for all methods
            from sparse_frontier.modelling.attention.registry import get_attention
            get_attention().sync_and_calc_layer_stats()

        if not is_prefilling:
            from sparse_frontier.modelling.attention.registry import get_attention

            get_attention().decode(
                query=queries,
                keys=keys,
                values=values,
                k_cache=k_cache,
                v_cache=v_cache,
                cache_seqlens=self.tokens_per_layer_head[self.current_layer],
                output=output,
                layer_idx=self.current_layer,
            )

        self.current_layer = (self.current_layer + 1) % self.model_layers
