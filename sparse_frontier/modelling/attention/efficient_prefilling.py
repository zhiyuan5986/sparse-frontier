import math
import torch
from typing import Optional, List, Tuple
from .abstract_attention import AbstractAttention
from .minference import (
    block_sparse_attention,
    vertical_and_slash_kernel,
    vertical_slash_sparse_attention,
    sum_over_diagonals,
)
from .abstract_attention import AttentionUtils
from .minference.vertical_and_slash import _triton_mixed_sparse_attention
from .minference.beacon import convert_beacon_indexes

class DenseAttention(AbstractAttention):
    """Standard dense attention with causal masking."""
    
    def __init__(self):
        super().__init__()

    def __call__(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        self.layer_sparsity_statistics.append(torch.tensor(0.0, device=queries.device))
        return AttentionUtils.flash_attention(queries, keys, values)


class BlockSparseAttentionMInference(AbstractAttention):
    """Block-sparse attention with chunk-level relevance and local windows."""
    
    def __init__(self, chunk_size: int, top_chunks: int):
        super().__init__()
        self.chunk_size = chunk_size
        self.top_chunks = top_chunks
        
        assert chunk_size >= 8, "Recommended chunk size is >= 8"
        assert top_chunks >= 3, "Must select at least one top chunk"

    @staticmethod
    def _calculate_sparsity(seq_len: int, chunk_size: int, top_chunks: int) -> float:
        """Calculate sparsity ratio based on selected blocks vs total possible blocks.
        
        For autoregressive attention:
        - First query block picks 1 key block
        - First top_chunks query blocks pick max possible key blocks
        - Remaining query blocks pick top_chunks key blocks each
        
        Args:
            seq_len: Length of input sequence
            chunk_size: Size of each attention block
            top_chunks: Number of chunks to select per query
            
        Returns:
            Sparsity ratio between 0 and 1
        """
        num_blocks = (seq_len + chunk_size - 1) // chunk_size

        total_blocks = num_blocks * (num_blocks + 1) // 2
        
        selected_blocks = top_chunks * (top_chunks + 1) // 2
        selected_blocks += (num_blocks - top_chunks) * top_chunks
        
        return 1.0 - (selected_blocks / total_blocks)

    @staticmethod
    def _get_blocks_for_sparsity(seq_len: int, chunk_size: int, target_sparsity: float) -> int:
        """Calculate number of blocks needed to achieve desired sparsity level.
        
        Uses binary search to find the number of blocks that gives sparsity closest
        to the target. The relationship between blocks and sparsity is monotonic.
        
        Args:
            seq_len: Length of input sequence
            chunk_size: Size of each attention block
            target_sparsity: Desired sparsity ratio between 0 and 1
            
        Returns:
            Number of blocks to select per query to achieve target sparsity
        """
        num_blocks = (seq_len + chunk_size - 1) // chunk_size
        
        # Binary search for number of blocks
        left, right = 3, num_blocks  # Minimum 4 blocks needed
        best_blocks = 3
        best_diff = float('inf')
        
        while left <= right:
            mid = (left + right) // 2
            sparsity = BlockSparseAttentionMInference._calculate_sparsity(seq_len, chunk_size, mid)
            
            diff = abs(sparsity - target_sparsity)
            if diff < best_diff:
                best_diff = diff
                best_blocks = mid
                
            if sparsity < target_sparsity:
                right = mid - 1
            else:
                left = mid + 1
                
        return best_blocks

    def __call__(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        sparsity = torch.tensor(self._calculate_sparsity(queries.shape[2], self.chunk_size, self.top_chunks), device=queries.device)
        self.layer_sparsity_statistics.append(sparsity)
        return block_sparse_attention(queries, keys, values, self.top_chunks, self.chunk_size, self.chunk_size)


class VerticalAndSlashAttentionMInference(AbstractAttention):
    """Combines vertical and diagonal patterns for efficient sparse attention.
    
    Implements the Vertical and Slash attention mechanism that selects important tokens based on:
    1) Vertical patterns - Top-k tokens that receive high attention across all queries
    2) Diagonal patterns - Diagonal stripes that capture local dependencies
    """

    def __init__(
        self,
        vertical_size: int = 64,
        slash_size: int = 128,
        approximation_size: int = 64,
    ):
        """Initialize Vertical and Slash attention.

        Args:
            vertical_size (int): Number of vertical tokens to select
            slash_size (int): Number of diagonal stripes to select
        """
        super().__init__()
        self.vertical_size = vertical_size
        self.slash_size = slash_size
        self.approximation_size = approximation_size

    def __call__(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        attn_output, sparsity = vertical_and_slash_kernel(
            queries,
            keys,
            values,
            vertical_size=min(self.vertical_size, queries.shape[2]),
            slash_size=min(self.slash_size, queries.shape[2]),
            last_q=min(self.approximation_size, queries.shape[2]),
        )
        self.layer_sparsity_statistics.append(sparsity)
        return attn_output


class FlexPrefill(AbstractAttention):
    def __init__(
        self,
        alpha: float = 0.9,
        approximation_size: int = 512,
        min_budget: int = 256,
    ):
        super().__init__()
        self.alpha = alpha
        self.approximation_size = approximation_size
        self.min_budget = min_budget
        self.causal_mask = None

    @staticmethod
    def score_cover_topk(x: torch.Tensor, score: float):
        cumsum_x = torch.cumsum(torch.sort(x, dim=-1, descending=True).values, dim=-1)
        topk = torch.sum(cumsum_x <= score, dim=-1) + 1
        return topk

    def get_active_blocks(
        self, q, k, v
    ):
        _, _, seq_len, head_dim = q.shape

        # Compute attention scores for last queries
        last_q_tokens = q[..., -self.approximation_size:, :] / math.sqrt(head_dim)
        qk = torch.einsum('bhik,bhjk->bhij', last_q_tokens, k)

        # Apply causal masking
        if self.causal_mask is None:
            self.causal_mask = torch.arange(0, self.approximation_size, device=last_q_tokens.device)
            self.causal_mask = self.causal_mask[:, None] >= self.causal_mask[None, :]
            self.causal_mask = self.causal_mask[None, None, ...]
        
        qk[..., -self.approximation_size:] = torch.where(
            self.causal_mask,
            qk[..., -self.approximation_size:],
            torch.tensor(float("-inf"), device=qk.device, dtype=qk.dtype)
        )

        # Get attention patterns
        scores = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32)

        # Compute vertical patterns
        vertical = scores.mean(dim=-2)
        vertical_size = max(self.min_budget, self.score_cover_topk(vertical, self.alpha).item())
        vertical[..., :4] = float("inf")
        vertical_topk = torch.topk(vertical, vertical_size, -1).indices
        
        # Fixed local window for slash patterns
        slashes = sum_over_diagonals(scores)[..., :-self.approximation_size + 1] / self.approximation_size
        slash_size = max(self.min_budget, self.score_cover_topk(slashes, self.alpha).item())
        slashes[..., -64:] = float("inf")
        slash = (seq_len - 1) - torch.topk(slashes, slash_size, -1).indices

        return vertical_topk, slash

    def __call__(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        assert queries.shape[-1] == keys.shape[-1]

        # Get active blocks for sparse portion
        vertical_idx, slash_idx = self.get_active_blocks(queries, keys, values)

        # Calculate sparse attention for non-dense portion
        sparse_out, sparsity = vertical_slash_sparse_attention(
            queries,
            keys,
            values,
            vertical_idx,
            slash_idx,
        )

        self.layer_sparsity_statistics.append(sparsity)
        
        return sparse_out

# class BeaconPrefill(AbstractAttention):
#     """
#     Efficient prefill attention with 3-class visibility (super/global, beacon, normal).
#     - NO approximation_size: compute exactly for the Q queries and K keys given to this call.
#     - Visibility strictly matches training-time CHADataCollator.make_segment_mask rules:

#       Let segment_ids: 0 => global super, >0 => normal segment; is_beacon: bool.
#       For query i (global index) and key j (global index):
#         • Normal i:  can see same-seg past (normal+beacon, j<=i) + previous super (j<i)
#         • Beacon i:  can see same-seg past (normal+beacon, j<=i) + previous super (j<i)
#         • Global i:  can see previous super (including beacon), causal (j<=i)

#     Integration note:
#       In vLLM prefill, keys usually cover indices [0..K-1], queries are the tail [T-Q..T-1].
#       This module expects you've sliced q/k/v that way before calling.
#     """

#     def __init__(self):
#         super().__init__()
#         self.segment_ids: torch.Tensor | None = None   # Long [B, T]
#         self.is_beacon:   torch.Tensor | None = None   # Bool [B, T]

#     def set_token_meta(self, segment_ids: torch.Tensor, is_beacon: torch.Tensor):
#         """
#         segment_ids: LongTensor [B, T]  (0=global super, >0=normal segment)
#         is_beacon:   BoolTensor [B, T]
#         """
#         assert segment_ids.dim() == 2 and is_beacon.dim() == 2, \
#             "Expect [B,T] for segment_ids & is_beacon"
#         assert segment_ids.shape == is_beacon.shape
#         self.segment_ids = segment_ids
#         self.is_beacon = is_beacon.bool()

#     @staticmethod
#     def _build_prefill_bias(
#         segment_ids: torch.Tensor,    # [B, T]
#         is_beacon: torch.Tensor,      # [B, T] bool
#         num_q: int,                   # Q
#         num_k: int,                   # K
#         dtype: torch.dtype,
#         device: torch.device,
#     ) -> torch.Tensor:
#         """
#         Return attention bias for current prefill call: [B, Q, K], values in {0, -inf}.
#         Uses global indexing: queries map to i ∈ [T-Q..T-1], keys map to j ∈ [0..K-1].
#         """
#         B, T = segment_ids.shape
#         MINF = float("-inf")

#         # Global indices for this prefill
#         i_idx = torch.arange(T - num_q, T, device=device)  # [Q]
#         j_idx = torch.arange(0, num_k, device=device)      # [K]

#         seg = segment_ids.to(device)
#         bec = is_beacon.to(device).bool()
#         is_global_super = (seg == 0)
#         is_super = bec | is_global_super
#         is_token = (seg > 0) & (~bec)

#         seg_i = seg[:, i_idx]                       # [B,Q]
#         seg_j = seg[:, j_idx]                       # [B,K]
#         same_seg = (seg_i.unsqueeze(-1) == seg_j.unsqueeze(-2)) & (seg_i.unsqueeze(-1) > 0)  # [B,Q,K]

#         is_token_i  = is_token[:, i_idx].unsqueeze(-1)         # [B,Q,1]
#         is_token_j  = is_token[:, j_idx].unsqueeze(-2)         # [B,1,K]
#         is_beacon_i = bec[:, i_idx].unsqueeze(-1)              # [B,Q,1]
#         is_beacon_j = bec[:, j_idx].unsqueeze(-2)              # [B,1,K]
#         is_super_i  = is_super[:, i_idx].unsqueeze(-1)         # [B,Q,1]
#         is_super_j  = is_super[:, j_idx].unsqueeze(-2)         # [B,1,K]
#         is_global_super_i = is_global_super[:, i_idx].unsqueeze(-1)  # [B,Q,1]

#         # Causality via global indices
#         i_mat = i_idx.view(1, -1, 1)  # [1,Q,1]
#         j_mat = j_idx.view(1, 1, -1)  # [1,1,K]
#         causal_le = (j_mat <= i_mat)  # [1,Q,K]
#         causal_lt = (j_mat <  i_mat)  # [1,Q,K]

#         # Rules (exactly as in training)
#         token_to_seg    = is_token_i  & same_seg & causal_le & (is_token_j | is_beacon_j)
#         token_to_super  = is_token_i  & is_super_j & causal_lt

#         beacon_to_seg   = is_beacon_i & same_seg & causal_le & (is_token_j | is_beacon_j)
#         beacon_to_super = is_beacon_i & is_super_j & causal_lt

#         super_to_super  = is_global_super_i & is_super_j & causal_le

#         visible = token_to_seg | token_to_super | beacon_to_seg | beacon_to_super | super_to_super  # [B,Q,K]

#         # Bias: 0 for visible, -inf otherwise
#         bias = torch.full((B, num_q, num_k), MINF, dtype=dtype, device=device)
#         bias.masked_fill_(visible, 0.0)
#         return bias

#     def __call__(
#         self,
#         queries: torch.Tensor,  # [B, H, Q, Dh]  <- prefill queries (tail)
#         keys: torch.Tensor,     # [B, H, K, Dh]  <- prefill keys   (prefix)
#         values: torch.Tensor,   # [B, H, K, Dh]
#         layer_idx: int,
#     ) -> torch.Tensor:
#         """
#         Return: [B, H, Q, Dh] (outputs only for current prefill queries)
#         """
#         assert self.segment_ids is not None and self.is_beacon is not None, \
#             "Call set_token_meta([B,T],[B,T]) before forward."

#         B, H, Q, Dh = queries.shape
#         _, _, K, _  = keys.shape
#         device = queries.device
#         dtype  = queries.dtype

#         # 1) Build [B,Q,K] attention bias (0/-inf)
#         bias = self._build_prefill_bias(
#             self.segment_ids, self.is_beacon, Q, K, dtype=dtype, device=device
#         )  # [B,Q,K]

#         # 2) Scaled dot-product on (Q,K)
#         scale = 1.0 / math.sqrt(Dh)
#         # logits: [B,H,Q,K]
#         logits = torch.einsum("bhiq,bhjk->bhij", queries, keys) * scale

#         # 3) Apply bias
#         logits = logits + bias.unsqueeze(1)  # [B,1,Q,K] -> broadcast over heads

#         # 4) Softmax and aggregate
#         attn = torch.softmax(logits, dim=-1, dtype=torch.float32).to(dtype)  # [B,H,Q,K]
#         out  = torch.matmul(attn, values)                                     # [B,H,Q,Dh]

#         # 5) Sparsity stat: invisible ratio over [B,Q,K]
#         vis_ratio = (bias == 0).float().mean()
#         sparsity  = 1.0 - vis_ratio
#         self.layer_sparsity_statistics.append(sparsity.detach().to(device))

#         return out

# class BeaconPrefill(AbstractAttention):
#     """
#     Efficient prefill attention with 3-class visibility (super/global, beacon, normal).
#     - 修改点仅在 prefill：只对本轮新增 queries（Q 行）与“同段窗口 + super 列”的紧凑 K/V 子集计算注意力。
#     - decode 阶段请沿用默认逻辑（本类不强制改写）。

#     约定：
#       queries: [B, H, Q, Dh]   —— 本轮 prefill 的新增查询（通常是序列尾部 Q 行）
#       keys:    [B, H, T, Dh]   —— 累积到当前的所有 K
#       values:  [B, H, T, Dh]   —— 累积到当前的所有 V

#     训练一致的可见性规则：
#       segment_ids: 0 => global super, >0 => 普通段
#       is_beacon:   True 表示 beacon
#       对于 query i 与 key j（均为全局索引）：
#         • 普通 i：  同段过去(普通+beacon, j<=i)  +  之前 super(j<i)
#         • beacon i：同段过去(普通+beacon, j<=i)  +  之前 super(j<i)
#         • super i： 之前所有 super(含 beacon, j<=i)
#     """

#     def __init__(self):
#         super().__init__()
#         self.segment_ids: torch.Tensor | None = None  # [B, T] Long
#         self.is_beacon:   torch.Tensor | None = None  # [B, T] Bool

#     # ===== 外部在进入 prefill 之前注入一次元信息 =====
#     def set_token_meta(self, segment_ids: torch.Tensor, is_beacon: torch.Tensor):
#         """
#         segment_ids: [B, T] Long  (0=global super, >0=normal segment)
#         is_beacon:   [B, T] Bool
#         """
#         assert segment_ids.dim() == 2 and is_beacon.dim() == 2, \
#             "Expect segment_ids/is_beacon with shape [B,T]."
#         assert segment_ids.shape == is_beacon.shape
#         self.segment_ids = segment_ids
#         self.is_beacon = is_beacon.bool()

#     # ======== 工具函数：当前段范围 / super 集合 / pack 选择 / 构造 bias ========
#     @staticmethod
#     @torch.no_grad()
#     def _current_segment_range(seg_1d: torch.Tensor) -> tuple[int, int]:
#         """返回最后一个 segment 的 [start, end) 区间；seg==0 时返回 (0,T)。"""
#         T = int(seg_1d.numel())
#         s_cur = int(seg_1d[-1].item())
#         if s_cur == 0:
#             return 0, T
#         p = T - 1
#         while p >= 0 and int(seg_1d[p].item()) == s_cur:
#             p -= 1
#         return p + 1, T

#     @staticmethod
#     @torch.no_grad()
#     def _super_indices(seg_1d: torch.Tensor, bec_1d: torch.Tensor) -> torch.Tensor:
#         """super 集合：seg==0 的 global 与所有 beacon（升序去重）。"""
#         sup = (seg_1d == 0) | bec_1d.bool()
#         idx = torch.nonzero(sup, as_tuple=False).flatten()
#         return torch.unique(idx, sorted=True)

#     @torch.no_grad()
#     def _build_prefill_pack(
#         self, seg_1d: torch.Tensor, bec_1d: torch.Tensor, Q: int
#     ):
#         """
#         返回 (j_idx, i_idx, (start,end))：
#           j_idx:  选中的列全局索引（同段窗口 + super，升序去重） [K_eff]
#           i_idx:  本轮 query 的全局行号（尾部 Q 行）          [Q]
#         """
#         T = int(seg_1d.numel())
#         device = seg_1d.device
#         i_idx = torch.arange(T - Q, T, device=device)  # 尾部 Q 行

#         s_cur = int(seg_1d[-1].item())
#         sup_idx = self._super_indices(seg_1d, bec_1d).to(device)

#         if s_cur == 0:
#             return sup_idx, i_idx, (0, 0)

#         start, end = self._current_segment_range(seg_1d)
#         win = torch.arange(start, end, device=device)
#         j_idx = torch.unique(torch.cat([win, sup_idx], dim=0), sorted=True)
#         return j_idx, i_idx, (start, end)

#     @staticmethod
#     @torch.no_grad()
#     def _make_bias_qk(
#         seg_1d: torch.Tensor,        # [T]
#         bec_1d: torch.Tensor,        # [T]
#         i_idx: torch.Tensor,         # [Q]
#         j_idx: torch.Tensor,         # [K_eff]
#         dtype: torch.dtype,
#     ) -> torch.Tensor:
#         """三类规则 + 因果，构造 [Q, K_eff] 的 0/-inf bias。"""
#         device = seg_1d.device
#         MINF = float("-inf")

#         seg = seg_1d
#         bec = bec_1d.bool()
#         is_global_super = (seg == 0)
#         is_super = bec | is_global_super
#         is_token = (seg > 0) & (~bec)

#         seg_i = seg[i_idx].view(-1, 1)
#         seg_j = seg[j_idx].view(1, -1)
#         same_seg = (seg_i == seg_j) & (seg_i > 0)

#         is_token_i  = is_token[i_idx].view(-1, 1)
#         is_token_j  = is_token[j_idx].view(1, -1)
#         is_beacon_i = bec[i_idx].view(-1, 1)
#         is_beacon_j = bec[j_idx].view(1, -1)
#         is_super_i  = is_super[i_idx].view(-1, 1)
#         is_super_j  = is_super[j_idx].view(1, -1)
#         is_global_super_i = is_global_super[i_idx].view(-1, 1)

#         i_mat = i_idx.view(-1, 1)
#         j_mat = j_idx.view(1, -1)
#         causal_le = (j_mat <= i_mat)
#         causal_lt = (j_mat <  i_mat)

#         token_to_seg    = is_token_i  & same_seg & causal_le & (is_token_j | is_beacon_j)
#         token_to_super  = is_token_i  & is_super_j & causal_lt

#         beacon_to_seg   = is_beacon_i & same_seg & causal_le & (is_token_j | is_beacon_j)
#         beacon_to_super = is_beacon_i & is_super_j & causal_lt

#         super_to_super  = is_global_super_i & is_super_j & causal_le

#         visible = token_to_seg | token_to_super | beacon_to_seg | beacon_to_super | super_to_super
#         bias = torch.full((i_idx.numel(), j_idx.numel()), MINF, dtype=dtype, device=device)
#         bias.masked_fill_(visible, 0.0)
#         return bias  # [Q, K_eff]

#     # ================= 主前向（仅用于 prefill） =================
#     def __call__(
#         self,
#         queries: torch.Tensor,  # [B, H, Q, Dh] —— 本轮新增 queries（尾部 Q）
#         keys: torch.Tensor,     # [B, H, T, Dh]
#         values: torch.Tensor,   # [B, H, T, Dh]
#         layer_idx: int,
#     ) -> torch.Tensor:
#         """
#         返回: [B, H, Q, Dh]

#         说明：
#         - 仅处理 prefill；decode 阶段请使用默认注意力/已有实现。
#         - 目前实现假设 B=1（与 vLLM 单请求 prefill 对齐）；多 batch 可外层循环。
#         """
#         assert self.segment_ids is not None and self.is_beacon is not None, \
#             "Call set_token_meta([B,T],[B,T]) before forward."
#         assert queries.size(0) == 1 == keys.size(0) == values.size(0), \
#             "This implementation assumes B=1; loop outside for multi-batch."

#         B, H, Q, Dh = queries.shape
#         _, _, T, _  = keys.shape
#         device = queries.device
#         dtype  = queries.dtype

#         seg_1d = self.segment_ids[0].to(device)   # [T]
#         bec_1d = self.is_beacon[0].to(device)     # [T]

#         # 1) 选列（同段窗口 + super），选行（尾部 Q）
#         j_idx, i_idx, _ = self._build_prefill_pack(seg_1d, bec_1d, Q)   # [K_eff], [Q]

#         # 2) 构建紧凑 K/V 子集
#         k_sub = keys[:, :, j_idx, :]     # [1,H,K_eff,Dh]
#         v_sub = values[:, :, j_idx, :]   # [1,H,K_eff,Dh]

#         # 3) 三类规则 bias（0/-inf, [Q,K_eff]）
#         bias = self._make_bias_qk(seg_1d, bec_1d, i_idx, j_idx, dtype=dtype)  # [Q,K_eff]

#         # 4) 小矩阵注意力计算
#         scale = 1.0 / math.sqrt(Dh)
#         # logits: [1,H,Q,K_eff]
#         logits = torch.einsum("bhiq,bhjk->bhij", queries, k_sub) * scale
#         logits = logits + bias.unsqueeze(0).unsqueeze(1)  # -> [1,H,Q,K_eff]

#         attn = torch.softmax(logits, dim=-1, dtype=torch.float32).to(dtype)  # [1,H,Q,K_eff]
#         out  = torch.matmul(attn, v_sub)                                     # [1,H,Q,Dh]

#         # 5) 统计稀疏率（可见比例的互补；用于监控）
#         vis_ratio = (bias == 0).float().mean()
#         sparsity  = 1.0 - vis_ratio
#         self.layer_sparsity_statistics.append(sparsity.detach().to(device))

#         return out

# 正式版
# class BeaconMixedSparsePrefill(AbstractAttention):
#     """
#     Prefill with beacon-style mixed sparsity:
#       • 每个 segment 作为一个“变宽块”，仅保留段内靠近对角线的局部依赖（可配 diag_window）
#       • 所有 super( seg==0 ) + beacon 作为竖直列，被后续所有行可见（按 i_max 做因果截断）

#     输入/输出同 efficient prefill 约定（同形）：
#       __call__(q,k,v,layer_idx):
#         q: [B, Hq,  T, Dh]
#         k: [B, Hkv, T, Dh]
#         v: [B, Hkv, T, Dh]
#       返回: [B, H, T, Dh]（H = Hq；如 Hkv < Hq 自动 repeat 对齐）

#     参数：
#       block_size_M / block_size_N : Triton 内核块大小
#       diag_window: int | None     : 段内局部窗口（以 token 计）；None 表示段内 causal full
#       auto_kv_repeat:             : 是否把 KV 头重复到 Hq 以适配内核（GQA）
#     """

#     def __init__(
#         self,
#         block_size_M: int = 64,
#         block_size_N: int = 64,
#         diag_window: int = None,
#         auto_kv_repeat: bool = True,
#     ):
#         super().__init__()
#         assert block_size_M in (1, 2, 4, 8, 16, 32, 64, 128)
#         assert block_size_N in (1, 2, 4, 8, 16, 32, 64, 128)
#         self.BM = block_size_M
#         self.BN = block_size_N
#         self.diag_window = diag_window  # None => 段内 causal full；否则只保留窗口宽度
#         self.auto_kv_repeat = auto_kv_repeat

#         self.segment_ids: torch.Tensor = None  # [B, T] Long
#         self.is_beacon:   torch.Tensor = None  # [B, T] Bool

#     # ---- 外部注入 token 元信息 ----
#     def set_token_meta(self, segment_ids: torch.Tensor, is_beacon: torch.Tensor):
#         assert segment_ids.dim() == 2 and is_beacon.dim() == 2 and segment_ids.shape == is_beacon.shape
#         self.segment_ids = segment_ids
#         self.is_beacon   = is_beacon.bool()

#     # ---- 将 [T] 的 segment_ids 切为连续段 ----
#     @staticmethod
#     @torch.no_grad()
#     def _segments(seg_1d: torch.Tensor) -> List[Tuple[int, int, int]]:
#         T = int(seg_1d.numel())
#         if T == 0: return []
#         changes = torch.nonzero(seg_1d[1:] != seg_1d[:-1], as_tuple=False).flatten() + 1
#         starts  = torch.cat([torch.tensor([0], device=seg_1d.device), changes])
#         ends    = torch.cat([changes, torch.tensor([T], device=seg_1d.device)])
#         return [(int(s.item()), int(e.item()), int(seg_1d[e-1].item())) for s,e in zip(starts, ends)]

#     # ---- 构造混合稀疏索引（变宽块 + 竖直列）----
#     @torch.no_grad()
#     def _build_mixed_sparse_indices(
#         self,
#         T: int,
#         seg_1d: torch.Tensor,
#         bec_1d: torch.Tensor,
#         device: torch.device,
#         H: int,
#     ):
#         """
#         对每个 row-block（大小 = BM）：
#           block 通路：覆盖 [left_bound(i_max) .. i_max] 的**完整 BN 对齐列块**
#                       其中 left_bound = max(seg_start(i_max), i_max - diag_window + 1)，
#                       若 diag_window=None 则 left_bound = seg_start(i_max)
#           column 通路：合并
#                        (a) 残缺前缀 [left_bound .. first_full_block_start-1]
#                        (b) super ∪ beacon（并按因果 col ≤ i_max 截断）
#         """
#         BM, BN = self.BM, self.BN
#         num_row_blks = (T + BM - 1) // BM

#         # 每个位置的“段起点”
#         seg_starts = torch.zeros(T, dtype=torch.int32, device=device)
#         for s, e, sid in self._segments(seg_1d):
#             seg_starts[s:e] = 0 if sid == 0 else s

#         # super ∪ beacon 的全集索引（升序）
#         sup_mask = (seg_1d == 0) | bec_1d.bool()
#         super_idx_all = torch.nonzero(sup_mask, as_tuple=False).flatten().to(torch.int32)

#         block_count_per_row = []
#         block_offsets_per_row = []
#         column_count_per_row = []
#         column_index_per_row = []

#         for rb in range(num_row_blks):
#             row_start = rb * BM
#             row_end   = min(T, (rb + 1) * BM)
#             i_max     = row_end - 1

#             cur_seg_id  = int(seg_1d[i_max].item())
#             seg_start_i = int(seg_starts[i_max].item())

#             # === 局部窗口左界 ===
#             if self.diag_window is None:
#                 left_bound = seg_start_i
#             else:
#                 left_bound = max(seg_start_i, i_max - self.diag_window + 1)

#             # ---- block 通路：仅放入完整 BN 对齐的列块 ----
#             if cur_seg_id == 0:
#                 blk_starts = torch.empty(0, dtype=torch.int32, device=device)
#             else:
#                 first_full = ((left_bound + BN - 1) // BN) * BN   # ceil 到 BN
#                 last_full  = (i_max // BN) * BN
#                 if last_full >= first_full:
#                     blk_starts = torch.arange(first_full, last_full + 1, BN,
#                                               device=device, dtype=torch.int32)
#                 else:
#                     blk_starts = torch.empty(0, dtype=torch.int32, device=device)

#             block_count_per_row.append(torch.tensor([blk_starts.numel()], dtype=torch.int32, device=device))
#             block_offsets_per_row.append(blk_starts)

#             # ---- column 通路：残缺前缀 + super/beacon（因果裁剪）----
#             # 残缺前缀：窗口左界到第一个完整块起点之前（若无完整块，则到 i_max）
#             prefix_cols = torch.empty(0, dtype=torch.int32, device=device)
#             if cur_seg_id != 0:
#                 first_full = ((left_bound + BN - 1) // BN) * BN
#                 prefix_right = min(first_full, i_max + 1)
#                 if left_bound < prefix_right:
#                     prefix_cols = torch.arange(left_bound, prefix_right,
#                                                device=device, dtype=torch.int32)

#             # super/beacon 纵列（因果 col ≤ i_max）
#             if super_idx_all.numel() > 0:
#                 sup_cols = super_idx_all[super_idx_all <= i_max]
#             else:
#                 sup_cols = torch.empty(0, dtype=torch.int32, device=device)

#             # 合并并去重
#             if prefix_cols.numel() > 0 and sup_cols.numel() > 0:
#                 cols = torch.unique(torch.cat([prefix_cols, sup_cols], dim=0), sorted=True)
#             elif prefix_cols.numel() > 0:
#                 cols = prefix_cols
#             else:
#                 cols = sup_cols

#             column_count_per_row.append(torch.tensor([cols.numel()], dtype=torch.int32, device=device))
#             column_index_per_row.append(cols)

#         # 固定形状封装
#         max_nnz_s = max((t.numel() for t in block_offsets_per_row), default=1)
#         max_nnz_v = max((t.numel() for t in column_index_per_row), default=1)

#         bc = torch.zeros((1, H, num_row_blks), dtype=torch.int32, device=device)
#         bo = torch.zeros((1, H, num_row_blks, max_nnz_s), dtype=torch.int32, device=device)
#         cc = torch.zeros((1, H, num_row_blks), dtype=torch.int32, device=device)
#         ci = torch.zeros((1, H, num_row_blks, max_nnz_v), dtype=torch.int32, device=device)

#         for rb in range(num_row_blks):
#             bc[:, :, rb] = block_count_per_row[rb]  # 所有 head 共享；必要时可 head-wise 定制
#             cc[:, :, rb] = column_count_per_row[rb]
#             if block_offsets_per_row[rb].numel() > 0:
#                 bo[:, :, rb, :block_offsets_per_row[rb].numel()] = block_offsets_per_row[rb]
#             if column_index_per_row[rb].numel() > 0:
#                 ci[:, :, rb, :column_index_per_row[rb].numel()] = column_index_per_row[rb]

#         return bc, bo, cc, ci

#     # --------------------------- 前向（prefill） ---------------------------
#     def __call__(
#         self,
#         queries: torch.Tensor,  # [B, Hq,  T, Dh]
#         keys: torch.Tensor,     # [B, Hkv, T, Dh]
#         values: torch.Tensor,   # [B, Hkv, T, Dh]
#         layer_idx: int,
#     ) -> torch.Tensor:
#         assert self.segment_ids is not None and self.is_beacon is not None, "call set_token_meta([B,T],[B,T]) first"
#         B, Hq, T, Dh = queries.shape
#         Bk, Hk, Tk, Dhk = keys.shape
#         assert B == Bk == 1, "当前实现假设 B=1；多 batch 请在外层循环"
#         assert Tk == T and Dhk == Dh

#         device = queries.device
#         seg_1d = self.segment_ids[0].to(device)
#         bec_1d = self.is_beacon[0].to(device)

#         # GQA：把 KV 头扩展到 Hq
#         if self.auto_kv_repeat and Hk != Hq:
#             assert Hq % Hk == 0, f"Hq({Hq}) 必须是 Hkv({Hk}) 的整数倍"
#             rep = Hq // Hk
#             keys   = keys.repeat_interleave(rep, dim=1)
#             values = values.repeat_interleave(rep, dim=1)
#             H = Hq
#         else:
#             H = Hq
#             assert Hk == H, "当 auto_kv_repeat=False 时，要求 Hkv == Hq"

#         # head_dim 对齐到 2^k
#         Dh_eff = Dh
#         if Dh_eff not in (16, 32, 64, 128, 256, 512):
#             Dh_p2 = 1 << math.ceil(math.log2(Dh_eff))
#             pad_d = Dh_p2 - Dh_eff
#             queries = torch.nn.functional.pad(queries, [0, pad_d, 0, 0, 0, 0, 0, 0])
#             keys    = torch.nn.functional.pad(keys,    [0, pad_d, 0, 0, 0, 0, 0, 0])
#             values  = torch.nn.functional.pad(values,  [0, pad_d, 0, 0, 0, 0, 0, 0])
#             Dh_eff  = Dh_p2

#         # 行维按 BM 补齐
#         pad_t = (self.BM - (T & (self.BM - 1))) & (self.BM - 1)
#         if pad_t:
#             queries = torch.nn.functional.pad(queries, [0, 0, 0, pad_t, 0, 0, 0, 0])
#             keys    = torch.nn.functional.pad(keys,    [0, 0, 0, pad_t, 0, 0, 0, 0])
#             values  = torch.nn.functional.pad(values,  [0, 0, 0, pad_t, 0, 0, 0, 0])

#         # 构建混合稀疏索引（变宽块 + 竖直列）
#         bc, bo, cc, ci = self._build_mixed_sparse_indices(T, seg_1d, bec_1d, device, H)

#         seqlens = torch.tensor([T], dtype=torch.int32, device=device)
#         sm_scale = (Dh ** -0.5)

#         out = _triton_mixed_sparse_attention(
#             queries, keys, values, seqlens,
#             bc, bo, cc, ci,
#             sm_scale,
#             block_size_M=self.BM,
#             block_size_N=self.BN,
#         )  # [B,H,T_pad,Dh_eff]

#         out = out[:, :, :T, :Dh]
#         # 统计稀疏率（粗略估算）
#         block_cells  = bc.sum(dim=-1) * (self.BM * self.BN)   # [1,H]
#         column_cells = cc.sum(dim=-1) * self.BM               # [1,H]
#         total_cells  = T * (T + 1) // 2
#         sparsity     = 1.0 - ((block_cells + column_cells) / total_cells).mean()
#         self.layer_sparsity_statistics.append(sparsity.detach())
#         return out

# class BeaconMixedSparsePrefill(AbstractAttention):
#     """
#     Mixed-sparse prefill：
#       - 段内：近对角窗口（diag_window=None 则段内 causal full）→ 完整 BLOCK_N 列走 block path，左侧零头走 column path
#       - 竖直列：super(seg==0) ∪ beacon，逐 row-block 以 i_max 因果截断
#       - 关键：当 BLOCK_M >= 16 时，通过在 segment 之间插入“行 pad”使每个 row-block 只包含一个 segment（或仅 pad）
#               确保正确性；pad 行不参与任何注意力，最后丢弃。
#     """

#     def __init__(
#         self,
#         block_size_M: int = 16,     # >= 16
#         block_size_N: int = 64,
#         diag_window: Optional[int] = None,
#         auto_kv_repeat: bool = True,
#     ):
#         super().__init__()
#         assert block_size_M in (16, 32, 64, 128)
#         assert block_size_N in (32, 64, 128)
#         self.BM = block_size_M
#         self.BN = block_size_N
#         self.diag_window = diag_window
#         self.auto_kv_repeat = auto_kv_repeat

#         self.segment_ids: Optional[torch.Tensor] = None  # [B,T]
#         self.is_beacon:   Optional[torch.Tensor] = None  # [B,T]

#     def set_token_meta(self, segment_ids: torch.Tensor, is_beacon: torch.Tensor):
#         assert segment_ids.dim()==2 and is_beacon.dim()==2 and segment_ids.shape==is_beacon.shape
#         self.segment_ids = segment_ids
#         self.is_beacon   = is_beacon.bool()

#     # --------- 计算 segment 连续段 ----------
#     @staticmethod
#     def _segments(seg_1d: torch.Tensor) -> List[Tuple[int,int,int]]:
#         T = int(seg_1d.numel())
#         if T == 0: return []
#         changes = torch.nonzero(seg_1d[1:] != seg_1d[:-1], as_tuple=False).flatten() + 1
#         starts  = torch.cat([torch.tensor([0], device=seg_1d.device), changes])
#         ends    = torch.cat([changes, torch.tensor([T], device=seg_1d.device)])
#         return [(int(s.item()), int(e.item()), int(seg_1d[e-1].item())) for s,e in zip(starts, ends)]

#     # --------- 规划：在段与段之间插入 pad 行，使每个 row-block 不跨段 ----------
#     @torch.no_grad()
#     def _plan_row_padding(self, seg_1d: torch.Tensor) -> Tuple[torch.Tensor, List[Tuple[int,int,int]], torch.Tensor]:
#         """
#         返回：
#           pad_map: [T] -> 每个原始行前累计插入的 pad 行数（prefix 形式）
#           padded_spans: 列表[(p_s, p_e, sid)]，是每个 segment 在“pad 后时间轴”上的范围（左闭右开）
#           keep_rows: [T_padded]，布尔，哪些 padded 行对应真实行（pad 行为 False）
#         """
#         BM = self.BM
#         spans = self._segments(seg_1d)
#         T = int(seg_1d.numel())
#         device = seg_1d.device

#         pad_map = torch.zeros(T, dtype=torch.int32, device=device)
#         keep_rows_list = []
#         padded_spans: List[Tuple[int,int,int]] = []

#         cur_pad = 0
#         cur_row = 0
#         for (s, e, sid) in spans:
#             seg_len = e - s
#             # 如果当前 row-block 偏移不在 BM 边界，先插 pad 把它对齐
#             misalign = (cur_row % BM)
#             if misalign != 0:
#                 need = BM - misalign
#                 cur_pad += need
#                 # 标记这些 pad 行为 False
#                 keep_rows_list.extend([False] * need)
#                 cur_row += need
#             # 记录该段在 pad 后的范围
#             p_s = cur_row
#             p_e = p_s + seg_len
#             padded_spans.append((p_s, p_e, sid))
#             # 更新 pad_map：段内每个原行 i，在它之前累计 pad 行数为 cur_pad
#             pad_map[s:e] = cur_pad
#             # 段内真实行
#             keep_rows_list.extend([True] * seg_len)
#             cur_row = p_e
#         # 末尾不用对齐；只是 kernel 会看到 T_padded=cur_row
#         keep_rows = torch.tensor(keep_rows_list, dtype=torch.bool, device=device)
#         return pad_map, padded_spans, keep_rows

#     # --------- 构建混合稀疏索引（对齐后时间轴） ----------
#     @torch.no_grad()
#     def _build_indices_aligned(
#         self,
#         T: int,
#         seg_1d: torch.Tensor,
#         bec_1d: torch.Tensor,
#         device: torch.device,
#         H: int,
#     ):
#         """
#         在“pad 后时间轴”上构造四个索引张量：
#           - 行块：严格不跨段（由 _plan_row_padding 保证）
#           - 列集合：段内近对角窗口的完整 BLOCK_N 块（block path）
#                     + 残缺前缀 + super/beacon（column path）
#           - 因果：对每个 row-block 用 i_max（pad 后的行号）截断列（≤ i_max）
#         """
#         BM, BN = self.BM, self.BN

#         # 规划 row pad & 映射
#         pad_map, padded_spans, keep_rows = self._plan_row_padding(seg_1d)
#         T_padded = int(keep_rows.numel())

#         # super/beacon 的“原始列索引”集合（升序）
#         super_mask = (seg_1d == 0) | bec_1d
#         super_idx_all = torch.nonzero(super_mask, as_tuple=False).flatten()  # 原始时间轴

#         # 准备 [1,H,R] / [1,H,R,NNZ] 容器
#         num_row_blks = (T_padded + BM - 1) // BM
#         block_offsets_per_row = []
#         column_index_per_row  = []
#         block_count_per_row   = []
#         column_count_per_row  = []

#         # 一个小工具：把“原始列 j”映射到“pad 后列 j' = j + pad_map[j]”
#         def j_to_padded(j: torch.Tensor) -> torch.Tensor:
#             return (j + pad_map[j]).to(torch.int32)

#         # 遍历 row-block，注意：现在 row-block 按 pad 后划分，且不会跨段
#         for rb in range(num_row_blks):
#             row_start = rb * BM
#             row_end   = min(T_padded, (rb+1) * BM)
#             i_max_p   = row_end - 1  # pad 后行号

#             # 确定这个 row-block 对应的 segment（或全 pad）
#             # 若全 pad，则不做任何列（让这个块输出 0）
#             rows_mask = keep_rows[row_start:row_end]
#             if not rows_mask.any():
#                 # 空块
#                 block_count_per_row.append(torch.tensor([0], dtype=torch.int32, device=device))
#                 column_count_per_row.append(torch.tensor([0], dtype=torch.int32, device=device))
#                 block_offsets_per_row.append(torch.empty(0, dtype=torch.int32, device=device))
#                 column_index_per_row.append(torch.empty(0, dtype=torch.int32, device=device))
#                 continue

#             # 找到块内最后一个真实行的“原始行号 i”及其 segment
#             # i_p = pad 后行号；i = 原始行号
#             i_p = int((rows_mask.nonzero(as_tuple=False).max() + row_start).item())
#             # 反查原始行号 i：pad 后到原始的映射是 i = i_p - pad_map[i]，但 pad_map 依赖 i 本身。
#             # 简便做法：在这个块内所有真实行的 pad 后行号集合里，枚举原始行号候选 i 并挑与 i_p 匹配的那个：
#             real_rows = (keep_rows.nonzero(as_tuple=False).flatten()).tolist()  # 所有真实行的 pad 后行号
#             # 为避免 O(T) 查找，我们通过 padded_spans 推断：
#             sid = None
#             seg_start_orig = None
#             for (p_s, p_e, s_id) in padded_spans:
#                 if p_s <= i_p < p_e:
#                     sid = s_id
#                     # 该段的原始范围 [orig_s, orig_e)
#                     # 反推 orig_s：满足 p_s = orig_s + pad_map[orig_s]
#                     # 我们可以通过 seg_1d 找到该段原始 s/e
#                     break
#             # 用 seg_1d 找该段原始 [s,e)
#             # 注意：padded_spans 顺序与原段顺序一致，可用计数来找第 k 段
#             # 这里简单暴力一点：遍历 seg_1d 段
#             spans = self._segments(seg_1d)
#             for (s, e, s_id) in spans:
#                 if s_id == sid:
#                     # 找到与 (p_s,p_e) 对应的段；由于段顺序相同，第一个匹配 sid 的就是这段
#                     seg_start_orig, seg_end_orig = s, e
#                     break

#             # 计算块内 i_max 的“原始行号” i_max = i_p - pad_map[i] 的逆映射
#             # 用近似法：i_max 原始必在该段 [s,e) 且满足 i_max + pad_map[i_max] == i_p
#             # 在 [s,e) 里二分或线性找一个 i_max 使 i_max + pad_map[i_max] == i_p
#             s, e = seg_start_orig, seg_end_orig
#             candidates = torch.arange(s, e, device=device, dtype=torch.int32)
#             i_p_from_i = candidates + pad_map[candidates]
#             # 找最接近且不大于 i_p 的
#             pos = torch.searchsorted(i_p_from_i, torch.tensor(i_p, device=device, dtype=torch.int32), right=True) - 1
#             pos = int(max(0, min(int(pos.item()), candidates.numel()-1)))
#             i_max = int(candidates[pos].item())
#             i_max_p_check = int((i_max + pad_map[i_max]).item())
#             if i_max_p_check != i_p:
#                 # 退一步：用 argmin |i_p_from_i - i_p|
#                 pos = int(torch.argmin((i_p_from_i - i_p).abs()).item())
#                 i_max = int(candidates[pos].item())

#             # 段内窗口左界（针对这个 row-block 的行集合，取最小 left_bound，以覆盖块内所有行）
#             if self.diag_window is None or sid == 0:
#                 left_bound = s
#             else:
#                 # 对块内真实行 i，left_bound(i) = max(s, i - W + 1)，取最小值
#                 W = self.diag_window
#                 # 块内真实行（原始行号）集合：{ i | row_start<= i+pad_map[i] < row_end }
#                 # 枚举该段原始 i ∈ [s,e)
#                 i_all = torch.arange(s, e, device=device, dtype=torch.int32)
#                 i_p_all = i_all + pad_map[i_all]
#                 in_block = (i_p_all >= row_start) & (i_p_all < row_end)
#                 if in_block.any():
#                     i_blk = i_all[in_block]
#                     lb_all = torch.maximum(torch.full_like(i_blk, s), i_blk - (W - 1))
#                     left_bound = int(lb_all.min().item())
#                 else:
#                     left_bound = s  # 理论不会走到这里

#             # ---- block path：完整 BLOCK_N 列块（在“原始列号”上对齐）----
#             if sid == 0:
#                 blk_starts = torch.empty(0, dtype=torch.int32, device=device)
#             else:
#                 first_full = ((left_bound + self.BN - 1) // self.BN) * self.BN
#                 last_full  = (i_max // self.BN) * self.BN
#                 if last_full >= first_full:
#                     blk_starts = torch.arange(first_full, last_full + 1, self.BN,
#                                               device=device, dtype=torch.int32)
#                 else:
#                     blk_starts = torch.empty(0, dtype=torch.int32, device=device)

#             # ---- column path：残缺前缀（原始列）+ super/beacon（原始列）→ 再映射到 pad 后列并因果裁剪 ----
#             if sid == 0:
#                 prefix_cols_orig = torch.empty(0, dtype=torch.int32, device=device)
#             else:
#                 first_full = ((left_bound + self.BN - 1) // self.BN) * self.BN
#                 prefix_right = min(first_full, i_max + 1)
#                 if left_bound < prefix_right:
#                     prefix_cols_orig = torch.arange(left_bound, prefix_right, device=device, dtype=torch.int32)
#                 else:
#                     prefix_cols_orig = torch.empty(0, dtype=torch.int32, device=device)

#             # super/beacon 原始列 ≤ i_max
#             sup_cols_orig = super_idx_all[super_idx_all <= i_max] if super_idx_all.numel() > 0 \
#                             else torch.empty(0, dtype=torch.int32, device=device)

#             if prefix_cols_orig.numel() > 0 and sup_cols_orig.numel() > 0:
#                 cols_orig = torch.unique(torch.cat([prefix_cols_orig, sup_cols_orig], dim=0), sorted=True)
#             elif prefix_cols_orig.numel() > 0:
#                 cols_orig = prefix_cols_orig
#             else:
#                 cols_orig = sup_cols_orig

#             # 把“原始列号”映射到“pad 后列号”
#             cols_padded = j_to_padded(cols_orig)
#             # 因果截断到 ≤ i_max_p
#             cols_padded = cols_padded[cols_padded <= i_max_p]

#             # block 通路也要把“原始块起点”映射为“pad 后块起点”
#             blk_starts_padded = j_to_padded(blk_starts)

#             block_offsets_per_row.append(blk_starts_padded)
#             block_count_per_row.append(torch.tensor([blk_starts_padded.numel()], dtype=torch.int32, device=device))
#             column_index_per_row.append(cols_padded)
#             column_count_per_row.append(torch.tensor([cols_padded.numel()], dtype=torch.int32, device=device))

#         # 固定形状打包
#         R = num_row_blks
#         max_nnz_s = max((t.numel() for t in block_offsets_per_row), default=1)
#         max_nnz_v = max((t.numel() for t in column_index_per_row), default=1)
#         bc = torch.zeros((1, H, R), dtype=torch.int32, device=device)
#         bo = torch.zeros((1, H, R, max_nnz_s), dtype=torch.int32, device=device)
#         cc = torch.zeros((1, H, R), dtype=torch.int32, device=device)
#         ci = torch.zeros((1, H, R, max_nnz_v), dtype=torch.int32, device=device)
#         for rb in range(R):
#             bc[:, :, rb] = block_count_per_row[rb]
#             cc[:, :, rb] = column_count_per_row[rb]
#             if block_offsets_per_row[rb].numel() > 0:
#                 bo[:, :, rb, :block_offsets_per_row[rb].numel()] = block_offsets_per_row[rb]
#             if column_index_per_row[rb].numel() > 0:
#                 ci[:, :, rb, :column_index_per_row[rb].numel()] = column_index_per_row[rb]

#         T_padded = int(keep_rows.numel())
#         padded_to_orig = torch.full((T_padded,), -1, dtype=torch.int32, device=device)
#         # 对每个段的 pad 后区间 (p_s..p_e) 映射回原始 (s..e)
#         # 我们已有 seg_1d，按段计算 s..e
#         spans_orig = self._segments(seg_1d)  # [(s,e,sid)]
#         cur = 0
#         for (s, e, sid), (p_s, p_e, sid2) in zip(spans_orig, padded_spans):
#             assert sid == sid2
#             L = e - s
#             padded_to_orig[p_s:p_s+L] = torch.arange(s, e, device=device, dtype=torch.int32)

#         # 逐 row-block 生成 [BLOCK_M] 左界向量
#         row_left_bounds_list = []
#         BM = self.BM; W = self.diag_window
#         # 为快速查询 seg_start(i)
#         seg_start_idx = torch.zeros_like(seg_1d, dtype=torch.int32, device=device)
#         if seg_1d.numel() > 0:
#             chg = torch.nonzero(seg_1d[1:] != seg_1d[:-1], as_tuple=False).flatten() + 1
#             starts = torch.cat([torch.tensor([0], device=device), chg])
#             ends   = torch.cat([chg, torch.tensor([seg_1d.numel()], device=device)])
#             for s, e in zip(starts.tolist(), ends.tolist()):
#                 sid = int(seg_1d[e-1].item())
#                 seg_start_idx[s:e] = 0 if sid == 0 else s

#         for rb in range(num_row_blks):
#             row_start = rb * BM
#             row_end   = min(T_padded, (rb+1) * BM)
#             LB_vec = torch.zeros((BM,), dtype=torch.int32, device=device)  # 默认 0
#             for r in range(BM):
#                 i_p = row_start + r
#                 if i_p >= row_end:
#                     break
#                 if not keep_rows[i_p]:
#                     LB_vec[r] = 0
#                     continue
#                 i = int(padded_to_orig[i_p].item())             # 原始行号
#                 seg_id = int(seg_1d[i].item())
#                 s = int(seg_start_idx[i].item())                # 段起点(原始)
#                 if (W is None) or (seg_id == 0):
#                     LB_orig = s
#                 else:
#                     LB_orig = max(s, i - W + 1)
#                 LB_pad = int((LB_orig + pad_map[LB_orig]).item())
#                 LB_vec[r] = LB_pad
#             row_left_bounds_list.append(LB_vec)

#         # 打包 [1,H,NUM_ROWS,BLOCK_M]
#         R = num_row_blks
#         row_left_bounds = torch.stack(row_left_bounds_list, dim=0)        # [R, BM]
#         row_left_bounds = row_left_bounds.view(1, 1, R, BM).expand(1, H, R, BM).contiguous()

#         return bc, bo, cc, ci, keep_rows, row_left_bounds

#     # --------------- 前向 ---------------
#     @torch.no_grad()
#     def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_idx: int) -> torch.Tensor:
#         assert self.segment_ids is not None and self.is_beacon is not None
#         B, Hq, T, Dh = q.shape
#         assert B == 1, "当前实现假设 B=1；多 batch 外层循环"
#         _, Hk, Tk, Dhk = k.shape
#         assert Tk == T and Dhk == Dh
#         device = q.device

#         # GQA：把 KV 头扩到 Hq
#         if self.auto_kv_repeat and Hk != Hq:
#             assert Hq % Hk == 0
#             rep = Hq // Hk
#             k = k.repeat_interleave(rep, dim=1)
#             v = v.repeat_interleave(rep, dim=1)
#             H = Hq
#         else:
#             assert Hk == Hq
#             H = Hq

#         # head_dim pad 到 2^k
#         Dh_eff = Dh
#         if Dh_eff not in (16, 32, 64, 128, 256, 512):
#             Dh_p2 = 1 << math.ceil(math.log2(Dh_eff))
#             pad_d = Dh_p2 - Dh_eff
#             q = torch.nn.functional.pad(q, [0, pad_d, 0, 0, 0, 0, 0, 0])
#             k = torch.nn.functional.pad(k, [0, pad_d, 0, 0, 0, 0, 0, 0])
#             v = torch.nn.functional.pad(v, [0, pad_d, 0, 0, 0, 0, 0, 0])
#             Dh_eff = Dh_p2

#         seg_1d = self.segment_ids[0].to(device)
#         bec_1d = self.is_beacon[0].to(device)

#         # 构建对齐后的索引 & 保留行标记
#         bc, bo, cc, ci, keep_rows, row_left_bounds = self._build_indices_aligned(T, seg_1d, bec_1d, device, H)
#         T_padded = int(keep_rows.numel())

#         # 把 q/k/v 的行维 pad 到 T_padded（pad 行填 0；k/v 也 pad 但不会被引用）
#         pad_rows = T_padded - T
#         if pad_rows > 0:
#             q = torch.nn.functional.pad(q, [0, 0, 0, pad_rows, 0, 0, 0, 0])  # pad 行在末尾
#             k = torch.nn.functional.pad(k, [0, 0, 0, pad_rows, 0, 0, 0, 0])
#             v = torch.nn.functional.pad(v, [0, 0, 0, pad_rows, 0, 0, 0, 0])

#         # 运行内核（seqlens = T_padded）
#         seqlens = torch.tensor([T_padded], dtype=torch.int32, device=device)
#         sm_scale = Dh ** -0.5
#         out = _triton_mixed_sparse_attention(
#             q, k, v, seqlens,
#             bc, bo, cc, ci,
#             row_left_bounds,
#             sm_scale,
#             block_size_M=self.BM,
#             block_size_N=self.BN,
#         )  # [1,H,T_padded, Dh_eff]

#         # 丢弃 pad 行、pad 维度
#         out = out[:, :, :T_padded, :Dh]
#         out = out[:, :, keep_rows, :]
#         return out  # [1,H,T,Dh]

# class BeaconMixedSparsePrefill(AbstractAttention):
#     """
#     Mixed-sparse prefill with segment-aware diagonal + vertical pattern:
#       - 段内近对角窗口：对齐 BLOCK_N 的整块列走 block path，左侧零头走 column path
#       - 竖直列：super(seg==0) ∪ beacon，因果裁剪
#       - 段对齐：在相邻 segment 之间插入 pad 行，使任一 row-block(BLOCK_M) 不跨段
#       - 去重：column path 去掉所有已被 block path 覆盖的列（在“原始列号”域做差集）
#       - 逐行左界：kernel 两条通路均施加 cols >= LB[row]（配合 row_left_bounds）
#     兼容 GQA（自动把 KV 头 repeat 到 Q 头）。
#     仅在 prefill 使用；decode 保持默认全因果注意力。
#     """

#     def __init__(
#         self,
#         block_size_M: int = 16,     # >= 16
#         block_size_N: int = 64,     # 32/64/128 常用
#         diag_window: Optional[int] = None,  # None 表示段内 causal full
#         auto_kv_repeat: bool = True,
#     ):
#         super().__init__()
#         assert block_size_M in (16, 32, 64, 128)
#         assert block_size_N in (32, 64, 128)
#         self.BM = block_size_M
#         self.BN = block_size_N
#         self.diag_window = diag_window
#         self.auto_kv_repeat = auto_kv_repeat
#         # 元信息：由上层在进入 prefill 前设置
#         self.segment_ids: Optional[torch.Tensor] = None  # [B,T]
#         self.is_beacon:   Optional[torch.Tensor] = None  # [B,T]

#     # =============== 外部喂入 token 元信息 ===============
#     def set_token_meta(self, segment_ids: torch.Tensor, is_beacon: torch.Tensor):
#         assert segment_ids.dim() == 2 and is_beacon.dim() == 2
#         assert segment_ids.shape == is_beacon.shape
#         self.segment_ids = segment_ids
#         self.is_beacon = is_beacon.bool()

#     # =============== 工具：切段 ===============
#     @staticmethod
#     def _segments(seg_1d: torch.Tensor) -> List[Tuple[int, int, int]]:
#         """
#         返回按相等 segment_id 连续段的 (start,end,sid)，end 为开区间
#         """
#         T = int(seg_1d.numel())
#         if T == 0:
#             return []
#         changes = torch.nonzero(seg_1d[1:] != seg_1d[:-1], as_tuple=False).flatten() + 1
#         starts = torch.cat([torch.tensor([0], device=seg_1d.device), changes])
#         ends = torch.cat([changes, torch.tensor([T], device=seg_1d.device)])
#         return [(int(s.item()), int(e.item()), int(seg_1d[e - 1].item())) for s, e in zip(starts, ends)]

#     # =============== 核心：段对齐的行 padding 计划 ===============
#     @torch.no_grad()
#     def _plan_row_padding(self, seg_1d: torch.Tensor):
#         """
#         在段与段之间插入 pad 行，使每个 row-block 不跨段。
#         返回：
#           pad_map: [T] 原 -> 在其前插入 pad 行的“累计数量”
#           padded_spans: [(p_s, p_e, sid)] 段在 pad 后时间轴上的范围
#           keep_rows: [T_padded] 布尔，pad 后哪些行是真实行
#         """
#         BM = self.BM
#         spans = self._segments(seg_1d)
#         T = int(seg_1d.numel())
#         device = seg_1d.device

#         pad_map = torch.zeros(T, dtype=torch.int32, device=device)
#         keep_rows: List[bool] = []
#         padded_spans: List[Tuple[int, int, int]] = []

#         cur_pad = 0
#         cur_row = 0
#         for (s, e, sid) in spans:
#             seg_len = e - s
#             # 对齐当前 row 到 BM 边界
#             mis = cur_row % BM
#             if mis != 0:
#                 need = BM - mis
#                 cur_pad += need
#                 keep_rows.extend([False] * need)
#                 cur_row += need
#             # 放置真实行
#             p_s = cur_row
#             p_e = p_s + seg_len
#             padded_spans.append((p_s, p_e, sid))
#             pad_map[s:e] = cur_pad
#             keep_rows.extend([True] * seg_len)
#             cur_row = p_e

#         keep_rows = torch.tensor(keep_rows, dtype=torch.bool, device=device)
#         return pad_map, padded_spans, keep_rows

#     # =============== 构造各 row-block 的索引与逐行左界 ===============
#     @torch.no_grad()
#     def _build_indices_aligned(
#         self,
#         T: int,
#         seg_1d: torch.Tensor,
#         bec_1d: torch.Tensor,
#         device: torch.device,
#         H: int,
#     ):
#         """
#         产出：
#           bc:[1,H,R], bo:[1,H,R,NNZ_S], cc:[1,H,R], ci:[1,H,R,NNZ_V],
#           row_left_bounds:[1,H,R,BM], keep_rows:[T_padded]
#         """
#         BM, BN = self.BM, self.BN

#         # 规划行 padding
#         pad_map, padded_spans, keep_rows = self._plan_row_padding(seg_1d)
#         T_padded = int(keep_rows.numel())

#         # 原始→pad 后列映射
#         def j_to_padded(j: torch.Tensor) -> torch.Tensor:
#             return (j + pad_map[j]).to(torch.int32)

#         # super（seg==0）∪ beacon 的原始列索引（升序）
#         super_mask = (seg_1d == 0) | bec_1d
#         super_idx_all = torch.nonzero(super_mask, as_tuple=False).flatten().to(torch.int32)

#         # 准备：pad 后行号 → 原始行号（pad 行为 -1）
#         padded_to_orig = torch.full((T_padded,), -1, dtype=torch.int32, device=device)
#         spans_orig = self._segments(seg_1d)
#         for (s, e, sid), (p_s, p_e, sid2) in zip(spans_orig, padded_spans):
#             assert sid == sid2
#             L = e - s
#             padded_to_orig[p_s:p_s + L] = torch.arange(s, e, device=device, dtype=torch.int32)

#         # 快速查段起点（原始行号）
#         seg_start_idx = torch.zeros_like(seg_1d, dtype=torch.int32, device=device)
#         if seg_1d.numel() > 0:
#             chg = torch.nonzero(seg_1d[1:] != seg_1d[:-1], as_tuple=False).flatten() + 1
#             starts = torch.cat([torch.tensor([0], device=device), chg])
#             ends = torch.cat([chg, torch.tensor([seg_1d.numel()], device=device)])
#             for s, e in zip(starts.tolist(), ends.tolist()):
#                 sid = int(seg_1d[e - 1].item())
#                 seg_start_idx[s:e] = 0 if sid == 0 else s

#         # 遍历 row-block，构造 4 个索引及逐行左界
#         num_row_blks = (T_padded + BM - 1) // BM
#         block_offsets_per_row = []
#         column_index_per_row = []
#         block_count_per_row = []
#         column_count_per_row = []
#         row_left_bounds_rb = []  # [R, BM]

#         for rb in range(num_row_blks):
#             row_start = rb * BM
#             row_end = min(T_padded, (rb + 1) * BM)
#             rows_mask = keep_rows[row_start:row_end]

#             # 构造逐行左界（pad 后列号）
#             LB_vec = torch.zeros((BM,), dtype=torch.int32, device=device)
#             for r in range(BM):
#                 i_p = row_start + r
#                 if i_p >= row_end or not keep_rows[i_p]:
#                     LB_vec[r] = 0  # pad 行，不参与
#                     continue
#                 i = int(padded_to_orig[i_p].item())  # 原始行号
#                 sid = int(seg_1d[i].item())
#                 s = int(seg_start_idx[i].item())     # 该行所在段的原始起点
#                 if (self.diag_window is None) or (sid == 0):
#                     LB_orig = s
#                 else:
#                     LB_orig = max(s, i - (self.diag_window - 1))
#                 LB_vec[r] = int((LB_orig + pad_map[LB_orig]).item())
#             row_left_bounds_rb.append(LB_vec)

#             # 行块内若全是 pad，给空索引
#             if not rows_mask.any():
#                 block_count_per_row.append(torch.tensor([0], dtype=torch.int32, device=device))
#                 column_count_per_row.append(torch.tensor([0], dtype=torch.int32, device=device))
#                 block_offsets_per_row.append(torch.empty(0, dtype=torch.int32, device=device))
#                 column_index_per_row.append(torch.empty(0, dtype=torch.int32, device=device))
#                 continue

#             # 取块内最后一个真实行，确定 i_max / 段信息
#             i_p_max = int((rows_mask.nonzero(as_tuple=False).max() + row_start).item())
#             i_max = int(padded_to_orig[i_p_max].item())       # 原始行号
#             sid = int(seg_1d[i_max].item())
#             s = int(seg_start_idx[i_max].item())

#             # 段内窗口左界（对块内所有真实行的最小 left_bound）
#             if (self.diag_window is None) or (sid == 0):
#                 left_bound = s
#             else:
#                 # 块内所有真实原始行号集合
#                 # i belongs to block iff i + pad_map[i] in [row_start, row_end)
#                 all_i = torch.arange(s, i_max + 1, device=device, dtype=torch.int32)
#                 i_p_all = all_i + pad_map[all_i]
#                 in_block = (i_p_all >= row_start) & (i_p_all < row_end) & keep_rows[i_p_all]
#                 if in_block.any():
#                     i_blk = all_i[in_block]
#                     lb_all = torch.maximum(torch.full_like(i_blk, s), i_blk - (self.diag_window - 1))
#                     left_bound = int(lb_all.min().item())
#                 else:
#                     left_bound = s

#             # ===== block path：对齐块（原始列号域）=====
#             if sid == 0:
#                 blk_starts = torch.empty(0, dtype=torch.int32, device=device)
#             else:
#                 first_full = ((left_bound + BN - 1) // BN) * BN
#                 last_full = (i_max // BN) * BN
#                 if last_full >= first_full:
#                     blk_starts = torch.arange(first_full, last_full + 1, BN, device=device, dtype=torch.int32)
#                 else:
#                     blk_starts = torch.empty(0, dtype=torch.int32, device=device)

#             # 这些块真实覆盖的原始列集合（∩ [0, i_max]，减少无效加载）
#             if blk_starts.numel() > 0:
#                 covered_ranges = []
#                 # 上界 i_max+1 是为了半开区间遍历
#                 for sblk in blk_starts.tolist():
#                     eblk = min(sblk + BN, i_max + 1)
#                     if eblk > sblk:
#                         covered_ranges.append(torch.arange(sblk, eblk, device=device, dtype=torch.int32))
#                 if covered_ranges:
#                     covered_by_blocks = torch.unique(torch.cat(covered_ranges), sorted=True)
#                 else:
#                     covered_by_blocks = torch.empty(0, dtype=torch.int32, device=device)
#             else:
#                 covered_by_blocks = torch.empty(0, dtype=torch.int32, device=device)

#             # ===== column path：残缺前缀 + super/beacon（原始列号域），再去掉 block 覆盖 =====
#             # 残缺前缀： [left_bound .. first_full) ∩ [.. i_max]
#             if sid == 0:
#                 prefix_cols_orig = torch.empty(0, dtype=torch.int32, device=device)
#             else:
#                 first_full = ((left_bound + BN - 1) // BN) * BN
#                 prefix_right = min(first_full, i_max + 1)
#                 if left_bound < prefix_right:
#                     prefix_cols_orig = torch.arange(left_bound, prefix_right, device=device, dtype=torch.int32)
#                 else:
#                     prefix_cols_orig = torch.empty(0, dtype=torch.int32, device=device)

#             # 竖直列（super ∪ beacon）：原始 ≤ i_max
#             if super_idx_all.numel() > 0:
#                 sup_cols_orig = super_idx_all[super_idx_all <= i_max]
#             else:
#                 sup_cols_orig = torch.empty(0, dtype=torch.int32, device=device)

#             # 合并 + 去重
#             if prefix_cols_orig.numel() > 0 and sup_cols_orig.numel() > 0:
#                 cols_orig = torch.unique(torch.cat([prefix_cols_orig, sup_cols_orig], dim=0), sorted=True)
#             elif prefix_cols_orig.numel() > 0:
#                 cols_orig = prefix_cols_orig
#             else:
#                 cols_orig = sup_cols_orig

#             # ⚠️ 差集：去掉已被 block 覆盖的所有列（在原始列号域做）
#             if cols_orig.numel() > 0 and covered_by_blocks.numel() > 0:
#                 covered_mask = torch.zeros(i_max + 1, dtype=torch.bool, device=device)
#                 covered_mask[covered_by_blocks] = True
#                 cols_orig = cols_orig[~covered_mask[cols_orig]]

#             # 映射到 pad 后列号；再按 pad 后 i_max_p 因果裁剪（通常已满足）
#             i_max_p = int((i_max + pad_map[i_max]).item())
#             if cols_orig.numel() > 0:
#                 cols_padded = j_to_padded(cols_orig)
#                 cols_padded = cols_padded[cols_padded <= i_max_p]
#             else:
#                 cols_padded = torch.empty(0, dtype=torch.int32, device=device)

#             # 块起点映射到 pad 后
#             if blk_starts.numel() > 0:
#                 blk_starts_padded = j_to_padded(blk_starts)
#             else:
#                 blk_starts_padded = torch.empty(0, dtype=torch.int32, device=device)

#             # 收集
#             block_offsets_per_row.append(blk_starts_padded)
#             block_count_per_row.append(torch.tensor([blk_starts_padded.numel()], dtype=torch.int32, device=device))
#             column_index_per_row.append(cols_padded)
#             column_count_per_row.append(torch.tensor([cols_padded.numel()], dtype=torch.int32, device=device))

#         # 打包固定形状
#         R = num_row_blks
#         max_nnz_s = max((t.numel() for t in block_offsets_per_row), default=1)
#         max_nnz_v = max((t.numel() for t in column_index_per_row), default=1)

#         bc = torch.zeros((1, H, R), dtype=torch.int32, device=device)
#         bo = torch.zeros((1, H, R, max_nnz_s), dtype=torch.int32, device=device)
#         cc = torch.zeros((1, H, R), dtype=torch.int32, device=device)
#         ci = torch.zeros((1, H, R, max_nnz_v), dtype=torch.int32, device=device)
#         for rb in range(R):
#             bc[:, :, rb] = block_count_per_row[rb]
#             cc[:, :, rb] = column_count_per_row[rb]
#             if block_offsets_per_row[rb].numel() > 0:
#                 bo[:, :, rb, :block_offsets_per_row[rb].numel()] = block_offsets_per_row[rb]
#             if column_index_per_row[rb].numel() > 0:
#                 ci[:, :, rb, :column_index_per_row[rb].numel()] = column_index_per_row[rb]

#         # row_left_bounds: [1,H,R,BM]
#         row_left_bounds = torch.stack(row_left_bounds_rb, dim=0)  # [R,BM]
#         row_left_bounds = row_left_bounds.view(1, 1, R, BM).expand(1, H, R, BM).contiguous()

#         return bc, bo, cc, ci, row_left_bounds, keep_rows

#     # =============== 前向（prefill 部分） ===============
#     @torch.no_grad()
#     def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_idx: int) -> torch.Tensor:
#         """
#         q: [B, Hq, T, Dh], k/v: [B, Hkv, T, Dh]
#         返回：out_sparse 与 dense 等价（数值上仅有 bf16/fp16 微小差）
#         """
#         assert self.segment_ids is not None and self.is_beacon is not None, "segment/is_beacon 未设置"
#         B, Hq, T, Dh = q.shape
#         assert B == 1, "当前实现假定 B=1（vLLM prefill 通常如此）"
#         _, Hkv, Tk, Dhk = k.shape
#         assert Tk == T and Dhk == Dh
#         device = q.device

#         # GQA: KV 头 → Q 头
#         if self.auto_kv_repeat and Hkv != Hq:
#             assert Hq % Hkv == 0
#             rep = Hq // Hkv
#             k = k.repeat_interleave(rep, dim=1)
#             v = v.repeat_interleave(rep, dim=1)
#             H = Hq
#         else:
#             assert Hkv == Hq
#             H = Hq

#         # head_dim pad 到 2^k（Triton 对齐）
#         Dh_eff = Dh
#         if Dh_eff not in (16, 32, 64, 128, 256, 512):
#             Dh_p2 = 1 << math.ceil(math.log2(Dh_eff))
#             pad_d = Dh_p2 - Dh_eff
#             q = torch.nn.functional.pad(q, [0, pad_d, 0, 0, 0, 0, 0, 0])
#             k = torch.nn.functional.pad(k, [0, pad_d, 0, 0, 0, 0, 0, 0])
#             v = torch.nn.functional.pad(v, [0, pad_d, 0, 0, 0, 0, 0, 0])
#             Dh_eff = Dh_p2

#         seg_1d = self.segment_ids[0].to(device)
#         bec_1d = self.is_beacon[0].to(device)

#         # 构造索引 & 逐行左界 & keep_rows
#         bc, bo, cc, ci, row_left_bounds, keep_rows = self._build_indices_aligned(
#             T, seg_1d, bec_1d, device, H
#         )
#         T_padded = int(keep_rows.numel())

#         # 行 pad 到 T_padded（pad 行的 q/k/v = 0）
#         if T_padded > T:
#             pad_rows = T_padded - T
#             q = torch.nn.functional.pad(q, [0, 0, 0, pad_rows, 0, 0, 0, 0])
#             k = torch.nn.functional.pad(k, [0, 0, 0, pad_rows, 0, 0, 0, 0])
#             v = torch.nn.functional.pad(v, [0, 0, 0, pad_rows, 0, 0, 0, 0])

#         # 调 kernel（seqlen 用 T_padded）
#         seqlens = torch.tensor([T_padded], dtype=torch.int32, device=device)
#         sm_scale = Dh ** -0.5
#         out = _triton_mixed_sparse_attention(
#             q, k, v, seqlens,
#             bc, bo, cc, ci,
#             row_left_bounds,                     # ★ 逐行左界
#             sm_scale,
#             block_size_M=self.BM,
#             block_size_N=self.BN,
#         )  # [1, H, T_padded, Dh_eff]

#         # 剪去 pad 行与 pad 维度
#         out = out[:, :, :T_padded, :Dh]
#         out = out[:, :, keep_rows, :]  # 仅保留真实行
#         # 统计（可选）
#         # sparsity 统计可以基于 block/column 占比自行记录
#         return out

# class BeaconMixedSparsePrefill:
#     """
#     高效 Beacon 版 prefill：
#       - 每段 = 一条不等宽“斜带”；对齐块走 block path，残缺/竖直列走 column path
#       - 段间插入 pad 行，保证任一 row-block(BLOCK_M) 不跨段
#       - 列通路做“去重（差集）”，避免重复计算
#       - 若 kernel 未打“列通路逐行因果/左界补丁”，可开启 conservative_columns=True 避免越权

#     使用：
#       sp = BeaconSparsePrefill(BLOCK_M=16, BLOCK_N=64, diag_window=64, conservative_columns=True)
#       out = sp(q, k, v, segment_ids, is_beacon)
#     """
#     def __init__(
#         self,
#         BLOCK_M: int = 16,
#         BLOCK_N: int = 64,
#         diag_window: Optional[int] = 64,   # None = 段内 full causal
#         conservative_columns: bool = True, # 若未打内核补丁，建议 True
#         auto_kv_repeat: bool = True,
#     ):
#         assert BLOCK_M in (16, 32, 64, 128)
#         assert BLOCK_N in (32, 64, 128)
#         self.BM = BLOCK_M
#         self.BN = BLOCK_N
#         self.diag_window = diag_window
#         self.conservative_columns = conservative_columns
#         self.auto_kv_repeat = auto_kv_repeat

#     # -------- helpers --------
#     @staticmethod
#     def _segments(seg_1d: torch.Tensor) -> List[Tuple[int, int, int]]:
#         """把相等 seg_id 的连续段切出来，返回 (start, end, sid)（end 开区间）"""
#         T = int(seg_1d.numel())
#         if T == 0:
#             return []
#         chg = torch.nonzero(seg_1d[1:] != seg_1d[:-1], as_tuple=False).flatten() + 1
#         starts = torch.cat([torch.tensor([0], device=seg_1d.device), chg])
#         ends   = torch.cat([chg, torch.tensor([T], device=seg_1d.device)])
#         return [(int(s.item()), int(e.item()), int(seg_1d[e-1].item())) for s, e in zip(starts, ends)]

#     def _plan_row_padding(self, seg_1d: torch.Tensor):
#         """段间插 pad 行，保证 row-block 不跨段。返回 pad_map, padded_spans, keep_rows"""
#         BM = self.BM
#         spans = self._segments(seg_1d)
#         T = int(seg_1d.numel())
#         dev = seg_1d.device

#         pad_map = torch.zeros(T, dtype=torch.int32, device=dev)
#         keep_rows: List[bool] = []
#         padded_spans: List[Tuple[int,int,int]] = []

#         cur_pad = 0
#         cur_row = 0
#         for (s, e, sid) in spans:
#             seg_len = e - s
#             mis = cur_row % BM
#             if mis != 0:
#                 need = BM - mis
#                 keep_rows.extend([False]*need)
#                 cur_pad += need
#                 cur_row += need
#             p_s = cur_row
#             p_e = p_s + seg_len
#             padded_spans.append((p_s, p_e, sid))
#             pad_map[s:e] = cur_pad
#             keep_rows.extend([True]*seg_len)
#             cur_row = p_e

#         keep_rows = torch.tensor(keep_rows, dtype=torch.bool, device=dev)
#         return pad_map, padded_spans, keep_rows

#     # -------- index builder --------
#     @torch.no_grad()
#     def _build_indices(
#         self,
#         seg_1d: torch.Tensor,      # [T]
#         beacon_1d: torch.Tensor,   # [T]
#         H: int,
#     ):
#         """
#         产出：block_count/block_offset/column_count/column_index（[1,H,R,...]）和 keep_rows
#         """
#         dev = seg_1d.device
#         T   = int(seg_1d.numel())
#         BM, BN = self.BM, self.BN

#         # 1) 段对齐 padding（行轴）
#         pad_map, padded_spans, keep_rows = self._plan_row_padding(seg_1d)
#         T_pad = int(keep_rows.numel())

#         # 原->pad 的列映射
#         def to_pad_cols(j: torch.Tensor) -> torch.Tensor:
#             return (j + pad_map[j]).to(torch.int32)

#         # super/beacon 列集合（原始列号，升序）
#         super_mask = (seg_1d == 0) | beacon_1d
#         sup_cols_all = torch.nonzero(super_mask, as_tuple=False).flatten().to(torch.int32)

#         # pad 行反查原始行（pad 行=-1）
#         padded_to_orig = torch.full((T_pad,), -1, dtype=torch.int32, device=dev)
#         spans_orig = self._segments(seg_1d)
#         for (s, e, sid), (p_s, p_e, sid2) in zip(spans_orig, padded_spans):
#             assert sid == sid2
#             L = e - s
#             padded_to_orig[p_s:p_s+L] = torch.arange(s, e, device=dev, dtype=torch.int32)

#         # 快速查每个原始 i 的段起点
#         seg_start_idx = torch.zeros_like(seg_1d, dtype=torch.int32, device=dev)
#         if T > 0:
#             chg = torch.nonzero(seg_1d[1:] != seg_1d[:-1], as_tuple=False).flatten() + 1
#             starts = torch.cat([torch.tensor([0], device=dev), chg])
#             ends   = torch.cat([chg, torch.tensor([T], device=dev)])
#             for s, e in zip(starts.tolist(), ends.tolist()):
#                 sid = int(seg_1d[e-1].item())
#                 seg_start_idx[s:e] = 0 if sid == 0 else s

#         # 遍历每个 row-block 生成索引
#         R = (T_pad + BM - 1) // BM
#         bc_list, bo_list, cc_list, ci_list = [], [], [], []

#         for rb in range(R):
#             row_start = rb * BM
#             row_end   = min(T_pad, (rb+1)*BM)
#             rows_mask = keep_rows[row_start:row_end]

#             # 空块（全 pad）
#             if not rows_mask.any():
#                 bc_list.append(torch.tensor([0], dtype=torch.int32, device=dev))
#                 bo_list.append(torch.empty(0, dtype=torch.int32, device=dev))
#                 cc_list.append(torch.tensor([0], dtype=torch.int32, device=dev))
#                 ci_list.append(torch.empty(0, dtype=torch.int32, device=dev))
#                 continue

#             # 块内最后真实行（pad 后 & 原始）
#             i_p_max = int((rows_mask.nonzero(as_tuple=False).max() + row_start).item())
#             i_max   = int(padded_to_orig[i_p_max].item())
#             sid     = int(seg_1d[i_max].item())
#             s       = int(seg_start_idx[i_max].item())

#             # 近对角左界（本块内“最小”左界，保证块内所有行都覆盖）
#             if (self.diag_window is None) or (sid == 0):
#                 left_bound = s
#             else:
#                 # 找到块内所有真实原始行 i，取 min_i max(s, i - (W-1))
#                 all_i   = torch.arange(s, i_max+1, device=dev, dtype=torch.int32)
#                 i_p_all = all_i + pad_map[all_i]
#                 in_blk  = (i_p_all >= row_start) & (i_p_all < row_end) & keep_rows[i_p_all]
#                 if in_blk.any():
#                     i_blk = all_i[in_blk]
#                     lb_all = torch.maximum(torch.full_like(i_blk, s), i_blk - (self.diag_window - 1))
#                     left_bound = int(lb_all.min().item())
#                 else:
#                     left_bound = s

#             # ====== 块通路：对齐 BN 的主带块（原始列号域）======
#             if sid == 0:
#                 blk_starts = torch.empty(0, dtype=torch.int32, device=dev)
#             else:
#                 first_full = ((left_bound + BN - 1)//BN)*BN
#                 last_full  = (i_max // BN) * BN
#                 if last_full >= first_full:
#                     blk_starts = torch.arange(first_full, last_full+1, BN, device=dev, dtype=torch.int32)
#                 else:
#                     blk_starts = torch.empty(0, dtype=torch.int32, device=dev)

#             # 实际被这些块覆盖的列（∩ [0, i_max]），用于差集
#             if blk_starts.numel() > 0:
#                 covered = []
#                 for sblk in blk_starts.tolist():
#                     eblk = min(sblk + BN, i_max + 1)
#                     if eblk > sblk:
#                         covered.append(torch.arange(sblk, eblk, device=dev, dtype=torch.int32))
#                 covered_by_blocks = torch.unique(torch.cat(covered), sorted=True) if covered else \
#                                     torch.empty(0, dtype=torch.int32, device=dev)
#             else:
#                 covered_by_blocks = torch.empty(0, dtype=torch.int32, device=dev)

#             # ====== 列通路：前缀残缺 + 竖直列（原始列号域）======
#             # 残缺前缀：[left_bound .. first_full) ∩ [.. i_max]
#             if sid == 0:
#                 prefix_cols = torch.empty(0, dtype=torch.int32, device=dev)
#             else:
#                 first_full = ((left_bound + BN - 1)//BN)*BN
#                 right = min(first_full, i_max+1)
#                 prefix_cols = torch.arange(left_bound, right, device=dev, dtype=torch.int32) \
#                               if left_bound < right else torch.empty(0, dtype=torch.int32, device=dev)

#             # 竖直列：super ∪ beacon，≤ i_max（因果）
#             sup_cols = sup_cols_all[sup_cols_all <= i_max] if sup_cols_all.numel() > 0 \
#                        else torch.empty(0, dtype=torch.int32, device=dev)

#             # 合并 + 去重
#             if prefix_cols.numel() > 0 and sup_cols.numel() > 0:
#                 cols_orig = torch.unique(torch.cat([prefix_cols, sup_cols], dim=0), sorted=True)
#             elif prefix_cols.numel() > 0:
#                 cols_orig = prefix_cols
#             else:
#                 cols_orig = sup_cols

#             # ⚠️ 差集：去掉已被块通路覆盖的列
#             if cols_orig.numel() > 0 and covered_by_blocks.numel() > 0:
#                 covered_mask = torch.zeros(i_max + 1, dtype=torch.bool, device=dev)
#                 covered_mask[covered_by_blocks] = True
#                 cols_orig = cols_orig[~covered_mask[cols_orig]]

#             # （可选）保守模式：如果你的 kernel 还没有“列通路逐行因果”补丁，
#             # 则把列通路列统一裁到 <= 块内最早真实行 row_start_orig，避免越权。
#             if self.conservative_columns:
#                 # 找块内最早真实行的原始行号
#                 i_p_min = int((rows_mask.nonzero(as_tuple=False).min() + row_start).item())
#                 i_min   = int(padded_to_orig[i_p_min].item())
#                 cols_orig = cols_orig[cols_orig <= i_min]  # 防止块内早行越权

#             # pad 映射 & 最终剪裁（≤ i_max_p）
#             cols_pad = to_pad_cols(cols_orig) if cols_orig.numel() > 0 else \
#                        torch.empty(0, dtype=torch.int32, device=dev)
#             blk_pad  = to_pad_cols(blk_starts) if blk_starts.numel() > 0 else \
#                        torch.empty(0, dtype=torch.int32, device=dev)

#             # 收集
#             bc_list.append(torch.tensor([blk_pad.numel()], dtype=torch.int32, device=dev))
#             bo_list.append(blk_pad)
#             cc_list.append(torch.tensor([cols_pad.numel()], dtype=torch.int32, device=dev))
#             ci_list.append(cols_pad)

#         # 打包成固定形状（[1,H,R,...]）
#         max_nnz_s = max((t.numel() for t in bo_list), default=1)
#         max_nnz_v = max((t.numel() for t in ci_list), default=1)

#         bc = torch.zeros((1, H, R), dtype=torch.int32, device=dev)
#         bo = torch.zeros((1, H, R, max_nnz_s), dtype=torch.int32, device=dev)
#         cc = torch.zeros((1, H, R), dtype=torch.int32, device=dev)
#         ci = torch.zeros((1, H, R, max_nnz_v), dtype=torch.int32, device=dev)

#         for rb in range(R):
#             bc[0, :, rb] = bc_list[rb]            # 广播到所有 head
#             cc[0, :, rb] = cc_list[rb]
#             if bo_list[rb].numel() > 0:
#                 bo[0, :, rb, :bo_list[rb].numel()] = bo_list[rb]
#             if ci_list[rb].numel() > 0:
#                 ci[0, :, rb, :ci_list[rb].numel()] = ci_list[rb]

#         return bc, bo, cc, ci, keep_rows

#     # -------- forward（prefill） --------
#     @torch.no_grad()
#     def __call__(
#         self,
#         q: torch.Tensor,           # [B,Hq,T,Dh]
#         k: torch.Tensor,           # [B,Hk,T,Dh]
#         v: torch.Tensor,           # [B,Hk,T,Dh]
#         segment_ids: torch.Tensor, # [B,T]
#         is_beacon: torch.Tensor,   # [B,T]
#     ) -> torch.Tensor:
#         B, Hq, T, Dh = q.shape
#         assert B == 1, "prefill 通常 B=1，本实现目前要求 B=1"
#         _, Hk, Tk, Dhk = k.shape
#         assert Tk == T and Dhk == Dh
#         dev = q.device

#         # GQA：把 KV 头重复到 Q 头
#         if self.auto_kv_repeat and Hk != Hq:
#             assert Hq % Hk == 0
#             rep = Hq // Hk
#             k = k.repeat_interleave(rep, dim=1)
#             v = v.repeat_interleave(rep, dim=1)
#             H = Hq
#         else:
#             assert Hk == Hq
#             H = Hq

#         # Dh pad 到 2^k（对齐 Triton）
#         Dh_eff = Dh
#         if Dh_eff not in (16, 32, 64, 128, 256, 512):
#             Dh2 = 1 << math.ceil(math.log2(Dh_eff))
#             pad_d = Dh2 - Dh_eff
#             q = torch.nn.functional.pad(q, [0, pad_d, 0, 0, 0, 0, 0, 0])
#             k = torch.nn.functional.pad(k, [0, pad_d, 0, 0, 0, 0, 0, 0])
#             v = torch.nn.functional.pad(v, [0, pad_d, 0, 0, 0, 0, 0, 0])
#             Dh_eff = Dh2

#         seg_1d = segment_ids[0].to(dev)
#         bec_1d = is_beacon[0].to(dev).bool()

#         # 构造索引
#         bc, bo, cc, ci, keep_rows = self._build_indices(seg_1d, bec_1d, H)

#         # 行 pad（到 T_pad）
#         T_pad = int(keep_rows.numel())
#         if T_pad > T:
#             add = T_pad - T
#             q = torch.nn.functional.pad(q, [0, 0, 0, add, 0, 0, 0, 0])
#             k = torch.nn.functional.pad(k, [0, 0, 0, add, 0, 0, 0, 0])
#             v = torch.nn.functional.pad(v, [0, 0, 0, add, 0, 0, 0, 0])

#         # 调 Triton
#         seqlens = torch.tensor([T_pad], dtype=torch.int32, device=dev)
#         sm_scale = (Dh ** -0.5)
#         out = _triton_mixed_sparse_attention(
#             q, k, v, seqlens,
#             bc, bo, cc, ci,
#             sm_scale,
#             block_size_M=self.BM,
#             block_size_N=self.BN,
#         )  # [1,H,T_pad,Dh_eff]

#         # 去除 pad 行和 Dh pad
#         out = out[:, :, :T_pad, :Dh]
#         out = out[:, :, keep_rows, :]
#         return 


class BeaconMixedSparsePrefill:
    """
    高效 prefill：Segment 内 FULL（因果） + Vertical（super/beacon）
    - 行轴：按段对齐 padding，确保每个 row-block(BM) 不跨段（且 q/k/v 物理插入这些 pad 行）
    - 列轴：段内 FULL 用列块（BLOCK_N）覆盖；vertical（super/beacon 且在段首之前）走列通路
    - 在“pad 域”对列通路做差集，避免与块通路重叠
    - 仅用于 prefill；decode 走默认
    约束：当前实现假定 B=1
    """

    def __init__(self, block_size_M: int = 16, block_size_N: int = 64, auto_kv_repeat: bool = True):
        assert block_size_M in (16, 32, 64, 128)
        assert block_size_N in (32, 64, 128)
        self.BM = block_size_M
        self.BN = block_size_N
        self.auto_kv_repeat = auto_kv_repeat

    @staticmethod
    def _segments(seg_1d: torch.Tensor) -> List[Tuple[int, int, int]]:
        T = int(seg_1d.numel())
        if T == 0:
            return []
        chg = torch.nonzero(seg_1d[1:] != seg_1d[:-1], as_tuple=False).flatten() + 1
        starts = torch.cat([torch.tensor([0], device=seg_1d.device), chg])
        ends   = torch.cat([chg, torch.tensor([T], device=seg_1d.device)])
        return [(int(s.item()), int(e.item()), int(seg_1d[e - 1].item()))
                for s, e in zip(starts, ends)]

    def _plan_row_padding(self, seg_1d: torch.Tensor):
        """
        返回:
          pad_map:  [T] 原始行 -> 累计插入的 pad 行数
          padded_spans: [(p_s,p_e,seg_id)] pad 域的段区间
          keep_rows: [T_pad] Bool，True=真实行，False=插入的 pad 行
        """
        spans = self._segments(seg_1d)
        T = int(seg_1d.numel())
        pad_map = torch.zeros(T, dtype=torch.int32, device=seg_1d.device)
        keep_rows = []
        padded_spans = []
        cur_pad, cur_row = 0, 0
        for (s, e, sid) in spans:
            seg_len = e - s
            mis = cur_row % self.BM
            if mis != 0:
                need = self.BM - mis
                keep_rows.extend([False] * need)  # 插 pad 行把段顶到 row-block 边界
                cur_pad += need
                cur_row += need
            p_s = cur_row
            p_e = p_s + seg_len
            padded_spans.append((p_s, p_e, sid))
            pad_map[s:e] = cur_pad
            keep_rows.extend([True] * seg_len)
            cur_row = p_e
        keep_rows = torch.tensor(keep_rows, dtype=torch.bool, device=seg_1d.device)
        return pad_map, padded_spans, keep_rows

    # ---------- 关键实现：构造 mixed-sparse 索引 ----------
    def _build_full_and_vertical_indices(
        self,
        seg_1d: torch.Tensor,      # [T]
        beacon_1d: torch.Tensor,   # [T] bool
        H: int                     # 头数（把同一套索引复制到每个头）
    ):
        """
        返回:
          block_count:  [1, H, NUM_ROWS] (int32)
          block_offset: [1, H, NUM_ROWS, NNZ_S] (int32)  每块起点列( pad 域 )
          column_count:[1, H, NUM_ROWS] (int32)
          column_index:[1, H, NUM_ROWS, NNZ_V] (int32)   单列索引( pad 域 )
          keep_rows:    [T_pad] bool
          pad_map:      [T] int32  原域->pad 域行偏移
        """
        device = seg_1d.device
        T = seg_1d.numel()
        assert beacon_1d.shape == seg_1d.shape

        # 行轴 pad 计划
        pad_map, padded_spans, keep_rows = self._plan_row_padding(seg_1d)
        T_pad = int(keep_rows.numel())

        # 原域 -> pad 域 列映射（列的 pad 偏移与行一致）
        def to_pad_cols(cols_orig: torch.Tensor) -> torch.Tensor:
            return (cols_orig + pad_map[cols_orig]).to(torch.int32)

        # vertical 列（原域）：super 或 beacon
        vertical_orig = torch.nonzero(
            (seg_1d == 0) | (beacon_1d.bool()), as_tuple=False
        ).flatten().to(torch.int64)

        # pad 域行 -> 原域行（pad 行=-1）
        padded_to_orig = torch.full((T_pad,), -1, dtype=torch.int32, device=device)
        for (s, e, sid), (p_s, p_e, sid2) in zip(self._segments(seg_1d), padded_spans):
            L = e - s
            if L > 0:
                padded_to_orig[p_s:p_s + L] = torch.arange(s, e, dtype=torch.int32, device=device)

        # 每个原域行所在段的起点（原域）
        seg_start_idx = torch.empty(T, dtype=torch.int32, device=device)
        for s, e, sid in self._segments(seg_1d):
            seg_start_idx[s:e] = s

        # row-block 个数
        num_rows = (T_pad + self.BM - 1) // self.BM

        # 为每个 row-block 收集块/列
        all_blk_starts = []   # list of LongTensor (原域块起点)
        all_col_pad     = []  # list of IntTensor  (pad 域逐列)

        for rb in range(num_rows):
            r0 = rb * self.BM
            r1 = min(T_pad, (rb + 1) * self.BM)
            rows_mask = keep_rows[r0:r1]
            if not rows_mask.any():
                # 纯 pad 行块
                all_blk_starts.append(torch.empty(0, dtype=torch.int64, device=device))
                all_col_pad.append(torch.empty(0, dtype=torch.int32, device=device))
                continue

            # 这块中“最后一个真实行”的 pad 行号/原行号
            i_p_max = torch.nonzero(rows_mask, as_tuple=False).max().item() + r0
            i_max = int(padded_to_orig[i_p_max].item())          # 原域
            s_start = int(seg_start_idx[i_max].item())           # 原域

            # ===== 段内 FULL 可见范围（原域）: [s_start .. i_max] =====
            # 只选择“完全落在该范围内的整块”作为块通路
            blk_starts = []
            if s_start <= i_max:
                # 第一个完整 BN 块的起点 >= s_start
                first_blk = ((s_start + self.BN - 1) // self.BN) * self.BN
                last_blk  = (i_max // self.BN) * self.BN
                if last_blk >= first_blk:
                    # 只保留 (sblk + BN - 1) <= i_max 的整块
                    for sblk in range(first_blk, last_blk + 1, self.BN):
                        if (sblk >= s_start) and (sblk + self.BN - 1 <= i_max):
                            blk_starts.append(sblk)

            if len(blk_starts) > 0:
                blk_starts = torch.tensor(blk_starts, dtype=torch.int64, device=device)
            else:
                blk_starts = torch.empty(0, dtype=torch.int64, device=device)

            # 被块通路覆盖的列（原域，逐列集合）
            covered_by_blocks_orig = []
            for sblk in blk_starts.tolist():
                covered_by_blocks_orig.append(torch.arange(sblk, sblk + self.BN, dtype=torch.int64, device=device))
            covered_by_blocks_orig = torch.unique(
                torch.cat(covered_by_blocks_orig), sorted=True
            ) if len(covered_by_blocks_orig) > 0 else torch.empty(0, dtype=torch.int64, device=device)

            # 段内 residual 列（原域）：[s_start..i_max] 去掉 block 覆盖
            if s_start <= i_max:
                seg_needed = torch.arange(s_start, i_max + 1, dtype=torch.int64, device=device)
                if covered_by_blocks_orig.numel() > 0:
                    # 逐列级差集
                    mask = torch.ones_like(seg_needed, dtype=torch.bool)
                    if seg_needed.numel() > 0:
                        # 用集合差集（更直观）
                        covered_set = set(covered_by_blocks_orig.tolist())
                        keep_list = [int(x) for x in seg_needed.tolist() if x not in covered_set]
                        residual_seg_orig = torch.tensor(keep_list, dtype=torch.int64, device=device) \
                            if len(keep_list) > 0 else torch.empty(0, dtype=torch.int64, device=device)
                else:
                    residual_seg_orig = seg_needed
            else:
                residual_seg_orig = torch.empty(0, dtype=torch.int64, device=device)

            # 段首之前 vertical（原域），并满足因果 ≤ i_max
            if vertical_orig.numel() > 0:
                vertical_prior_orig = vertical_orig[(vertical_orig < s_start) & (vertical_orig <= i_max)]
            else:
                vertical_prior_orig = torch.empty(0, dtype=torch.int64, device=device)

            # 列通路候选（原域）
            cols_orig = torch.unique(
                torch.cat([residual_seg_orig, vertical_prior_orig]),
                sorted=True
            ) if (residual_seg_orig.numel() + vertical_prior_orig.numel()) > 0 else torch.empty(0, dtype=torch.int64, device=device)

            # 严格逐列去重：去掉会被块通路覆盖的列（在“pad 域”做差更安全）
            cols_pad = to_pad_cols(cols_orig) if cols_orig.numel() > 0 else torch.empty(0, dtype=torch.int32, device=device)
            if covered_by_blocks_orig.numel() > 0 and cols_pad.numel() > 0:
                covered_pad_cols = to_pad_cols(covered_by_blocks_orig)  # 逐列展开
                covered_set = set(covered_pad_cols.tolist())
                cols_pad = torch.tensor(
                    [int(x) for x in cols_pad.tolist() if x not in covered_set],
                    dtype=torch.int32, device=device
                )
            cols_pad = torch.unique(cols_pad, sorted=True) if cols_pad.numel() > 0 else cols_pad

            all_blk_starts.append(blk_starts)  # 原域块起点（后面统一转 pad 域）
            all_col_pad.append(cols_pad)       # 已是 pad 域逐列

        # 统计最大 nnz 以便打包张量
        max_nnz_s = max((x.numel() for x in all_blk_starts), default=0)
        max_nnz_v = max((x.numel() for x in all_col_pad),   default=0)
        if max_nnz_s == 0:
            max_nnz_s = 1
        if max_nnz_v == 0:
            max_nnz_v = 1

        block_count  = torch.zeros((1, H, num_rows), dtype=torch.int32, device=device)
        block_offset = torch.zeros((1, H, num_rows, max_nnz_s), dtype=torch.int32, device=device)
        column_count = torch.zeros((1, H, num_rows), dtype=torch.int32, device=device)
        column_index = torch.zeros((1, H, num_rows, max_nnz_v), dtype=torch.int32, device=device)

        # 打包：把原域块起点映射到 pad 域；列索引已是 pad 域
        for rb in range(num_rows):
            blk_st = all_blk_starts[rb]   # 原域
            col_pd = all_col_pad[rb]      # pad 域

            nb = int(blk_st.numel())
            nc = int(col_pd.numel())

            # 原域块起点 -> pad 域块起点
            if nb > 0:
                blk_pad = (blk_st + pad_map[blk_st]).to(torch.int32)  # 仍然是“块起点列”的 pad 坐标
            else:
                blk_pad = torch.empty(0, dtype=torch.int32, device=device)

            # 写入
            block_count[0, :, rb]  = nb
            column_count[0, :, rb] = nc
            if nb > 0:
                block_offset[0, :, rb, :nb] = blk_pad.unsqueeze(0).expand(H, nb)
            if nc > 0:
                column_index[0, :, rb, :nc] = col_pd.unsqueeze(0).expand(H, nc)
        print(block_count, block_offset, column_count, column_index, keep_rows, pad_map)
        return block_count, block_offset, column_count, column_index, keep_rows, pad_map



    @torch.no_grad()
    def __call__(
        self,
        q: torch.Tensor,            # [1, Hq, T, D]
        k: torch.Tensor,            # [1, Hk, T, D]
        v: torch.Tensor,            # [1, Hk, T, D]
        segment_ids: torch.Tensor,  # [1, T]
        is_beacon: torch.Tensor,    # [1, T] bool
    ) -> torch.Tensor:
        assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4
        B, Hq, T, Dh = q.shape
        assert B == 1, "当前实现仅支持 batch=1"
        _, Hk, Tk, Dhk = k.shape
        assert Tk == T and Dhk == Dh
        dev = q.device
        H = Hq

        # GQA：把 KV 头重复到 Q 头
        if self.auto_kv_repeat and Hk != Hq:
            assert Hq % Hk == 0
            rep = Hq // Hk
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        else:
            assert Hk == Hq

        # head_dim 对齐到 2 的幂（Triton 友好）
        Dh_eff = Dh
        if Dh_eff not in (16, 32, 64, 128, 256, 512):
            Dh2 = 1 << math.ceil(math.log2(Dh_eff))
            pad_d = Dh2 - Dh_eff
            q = torch.nn.functional.pad(q, [0, pad_d, 0, 0, 0, 0, 0, 0])
            k = torch.nn.functional.pad(k, [0, pad_d, 0, 0, 0, 0, 0, 0])
            v = torch.nn.functional.pad(v, [0, pad_d, 0, 0, 0, 0, 0, 0])
            Dh_eff = Dh2

        seg_1d = segment_ids[0].to(dev)
        bec_1d = is_beacon[0].to(dev).bool()

        # 构建混合稀疏索引，并拿到 keep_rows + pad_map
        block_count, block_offset, column_count, column_index, keep_rows, pad_map = \
            self._build_full_and_vertical_indices(seg_1d, bec_1d, H=H)

        # === 关键修正：按 pad_map 在“行轴”真实插入 pad 行，构造 q_pad/k_pad/v_pad ===
        T_pad = int(keep_rows.numel())
        q_pad = torch.zeros((1, H, T_pad, Dh_eff), dtype=q.dtype, device=dev)
        k_pad = torch.zeros_like(q_pad)
        v_pad = torch.zeros_like(q_pad)
        orig_idx = torch.arange(T, dtype=torch.int32, device=dev)
        pad_pos  = (orig_idx + pad_map[orig_idx]).long()  # 原始行 -> pad 后行
        q_pad[:, :, pad_pos, :] = q
        k_pad[:, :, pad_pos, :] = k
        v_pad[:, :, pad_pos, :] = v

        # seqlens 与张量一致
        seqlens  = torch.tensor([T_pad], dtype=torch.int32, device=dev)
        # sm_scale = Dh_eff ** -0.5
        sm_scale = 1.0 / math.sqrt(float(Dh)) # ★ 用原始 D 缩放

        # 一次 kernel：block + column 混合稀疏
        out = _triton_mixed_sparse_attention(
            q_pad, k_pad, v_pad, seqlens,
            block_count, block_offset, column_count, column_index,
            sm_scale, block_size_M=self.BM, block_size_N=self.BN,
        )  # [1, H, T_pad, Dh_eff]

        # 去掉 head_dim 与行轴 padding，只保留真实行
        out = out[:, :, keep_rows, :Dh]
        return out


# class BeaconPrefill(AbstractAttention):
#     """
#     Efficient prefill attention with segment-aware beacon sparsity:
#       - Segment内部使用全注意力
#       - Segment 0和beacon token可被后续所有token看到
#       - 其他segment内token仅在段内进行causal attention
#     """
    
#     def __init__(
#         self,
#         block_size_M: int = 64,
#         block_size_N: int = 64,
#     ):
#         super().__init__()
#         self.block_size_M = block_size_M
#         self.block_size_N = block_size_N
#         self.segment_ids = None  # [BATCH, N_CTX]
#         self.is_beacon = None    # [BATCH, N_CTX]

#     def set_meta_data(
#         self, 
#         segment_ids: torch.Tensor,  # [BATCH, N_CTX]
#         is_beacon: torch.Tensor     # [BATCH, N_CTX]
#     ):
#         """设置段ID和beacon标记"""
#         assert segment_ids.shape == is_beacon.shape, "segment_ids和is_beacon形状必须相同"
#         self.segment_ids = segment_ids
#         self.is_beacon = is_beacon.bool()

#     @staticmethod
#     def _get_segment_boundaries(segment_ids: torch.Tensor) -> List[Tuple[int, int, int]]:
#         """获取段边界信息: [(start, end, segment_id), ...]"""
#         T = segment_ids.numel()
#         if T == 0:
#             return []
            
#         changes = torch.where(segment_ids[1:] != segment_ids[:-1])[0] + 1
#         starts = torch.cat([torch.tensor([0], device=segment_ids.device), changes])
#         ends = torch.cat([changes, torch.tensor([T], device=segment_ids.device)])
        
#         return [(int(s.item()), int(e.item()), int(segment_ids[s].item())) 
#                 for s, e in zip(starts, ends)]

#     @staticmethod
#     def _calculate_sparsity(segment_ids: torch.Tensor, is_beacon: torch.Tensor) -> float:
#         """计算注意力矩阵的稀疏度"""
#         batch_size, seq_len = segment_ids.shape
#         total_cells = seq_len * (seq_len + 1) // 2  # 下三角矩阵元素总数
        
#         visible_cells = 0
#         for b in range(batch_size):
#             boundaries = BeaconPrefill._get_segment_boundaries(segment_ids[b])
#             beacons = torch.where(is_beacon[b])[0]
#             beacon_count = beacons.numel()
            
#             # 计算每个段的可见单元格
#             for (s, e, seg_id) in boundaries:
#                 seg_len = e - s
#                 seg_cells = seg_len * (seg_len + 1) // 2  # 段内三角矩阵
                
#                 if seg_id == 0:
#                     # 全局段可见所有之前的单元格
#                     visible_cells += seg_len * seq_len
#                 else:
#                     # 普通段仅可见段内单元格
#                     visible_cells += seg_cells
            
#             # 减去beacon重复计算的部分
#             visible_cells -= beacon_count * seq_len
        
#         return 1.0 - (visible_cells / (batch_size * total_cells))

#     def __call__(
#         self, 
#         queries: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
#         keys: torch.Tensor,     # [BATCH, N_HEADS, N_CTX, D_HEAD]
#         values: torch.Tensor,   # [BATCH, N_HEADS, N_CTX, D_HEAD]
#         layer_idx: int = None,
#     ) -> torch.Tensor:
#         assert self.segment_ids is not None and self.is_beacon is not None, \
#             "调用前请先使用set_meta_data设置segment_ids和is_beacon"
            
#         batch_size, num_heads, context_size, head_dim = queries.shape
#         device = queries.device
        
#         # 计算稀疏度
#         sparsity = self._calculate_sparsity(self.segment_ids, self.is_beacon)
#         self.layer_sparsity_statistics.append(torch.tensor(sparsity, device=device))
        
#         # 生成稀疏索引（传入num_heads确保维度匹配）
#         seqlens = torch.tensor([context_size] * batch_size, dtype=torch.int32, device=device)
#         sm_scale = head_dim ** -0.5
        
#         block_count, block_offset, column_count, column_index = convert_beacon_indexes(
#             seqlens, 
#             self.segment_ids, 
#             self.is_beacon,
#             context_size,
#             self.block_size_M,
#             self.block_size_N,
#             num_heads=num_heads,  # 新增：显式传入头数
#         )
        
#         # 使用混合稀疏注意力计算
#         out = _triton_mixed_sparse_attention(
#             queries, keys, values, seqlens,
#             block_count, block_offset, column_count, column_index,
#             sm_scale,
#             block_size_M=self.block_size_M,
#             block_size_N=self.block_size_N,
#         )
        
#         return out