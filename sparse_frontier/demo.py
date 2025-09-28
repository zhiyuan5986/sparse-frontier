# demo_beacon_mixed_sparse_check.py
import math
import torch

# === 导入你的类与混合稀疏内核包装 ===
from sparse_frontier.modelling.attention.efficient_prefilling import BeaconMixedSparsePrefill

# -----------------------------
# 参考：dense 版本（构造等价 mask）
# -----------------------------
@torch.no_grad()
def dense_reference_attention(q, k, v, segment_ids, is_beacon, diag_window=None):
    """
    q,k,v: [B,H,T,Dh]（已做 GQA 对齐）
    segment_ids: [B,T] ; is_beacon: [B,T]
    规则与 BeaconMixedSparsePrefill 对齐（默认 vertical 不限段首之前；如需限制，见下文注释）：
      - 每个 i 所在段的局部窗口：left_bound(i) = max(seg_start(i), i-diag_window+1)（diag_window=None 表示 seg_start(i) ）
      - 同段 j ∈ [left_bound(i) .. i] 可见（含因果 j<=i）
      - super := (seg==0) ∪ beacon 纵列对所有后续 i 可见（含因果 j<=i）
      - seg_i==0 时，没有段内块，只看 super（因果）
    """
    B, H, T, Dh = q.shape
    device = q.device
    j_idx = torch.arange(T, device=device)        # 1D 列索引

    seg = segment_ids
    bec = is_beacon.bool()
    super_mask = (seg == 0) | bec                 # [B,T]

    # 预计算每个位置所属段的起点（seg==0 给 0）
    seg_starts = torch.zeros_like(seg, dtype=torch.long)
    for b in range(B):
        s = seg[b]
        # 找连续段
        changes = torch.nonzero(s[1:] != s[:-1], as_tuple=False).flatten() + 1
        starts  = torch.cat([torch.tensor([0], device=device), changes])
        ends    = torch.cat([changes, torch.tensor([T], device=device)])
        for st, ed in zip(starts.tolist(), ends.tolist()):
            sid = int(s[ed - 1].item())
            seg_starts[b, st:ed] = st if sid != 0 else 0

    # 在 fp32 上构造 bias，更稳
    MINF = torch.finfo(torch.float32).min
    bias = torch.full((B, 1, T, T), MINF, dtype=torch.float32, device=device)

    for b in range(B):
        seg_b   = seg[b]           # [T]
        sup_b   = super_mask[b]    # [T]
        start_b = seg_starts[b]    # [T]
        for i in range(T):
            si = int(seg_b[i].item())
            causal = (j_idx <= i)                  # [T] 一维因果
            if si == 0:
                # super-only, causal
                visible_j = sup_b & causal
            else:
                seg_start_i = int(start_b[i].item())
                if diag_window is None:
                    left_bound = seg_start_i
                else:
                    left_bound = max(seg_start_i, i - diag_window + 1)
                in_seg  = (seg_b == si)           # [T]
                vis_seg = in_seg & (j_idx >= left_bound) & causal
                # 默认 vertical 不限段首（与你文档一致）：
                vis_sup = sup_b & causal
                # 若希望 vertical 只允许到段首之前，请改为：
                # vis_sup = sup_b & (j_idx < seg_start_i) & causal
                visible_j = vis_seg | vis_sup

            cols = torch.nonzero(visible_j, as_tuple=False).flatten()  # [K]
            bias[b, 0, i, cols] = 0.0
    print(bias)

    # 注意：在 fp32 上做 qk 与 softmax，再还原到输入 dtype
    scale = 1.0 / math.sqrt(float(Dh))
    logits = torch.einsum("bhid,bhjd->bhij", q.to(torch.float32), k.to(torch.float32)) * scale
    logits = logits + bias
    attn   = torch.softmax(logits, dim=-1)        # fp32
    out    = torch.matmul(attn, v.to(torch.float32)).to(q.dtype)
    return out  # [B,H,T,Dh]

# -----------------------------
# 造一个小例子做一致性校验
# -----------------------------
def run_demo():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 形状设定
    B, Hq, Hkv, T, Dh = 1, 8, 4, 100, 32   # Kv 是 GQA 的一半头
    diag_window = None                       # 近对角线窗口大小；可改成 None 验证段内 causal full

    # 构造 q/k/v
    q = torch.randn(B, Hq,  T, Dh, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, Hkv, T, Dh, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, Hkv, T, Dh, device=device, dtype=torch.bfloat16)

    # 构造 segment 与 beacon（与你之前的例子一致）
    # segment_ids = torch.tensor([[0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3, 0,0,0,0]], device=device, dtype=torch.long)
    segment_ids = torch.tensor([[0]*10+[1]*20+[2]*30+[3]*40], device=device, dtype=torch.long)
    is_beacon   = torch.zeros_like(segment_ids, dtype=torch.bool)
    # is_beacon[0,0]  = True   # seg=1 内 beacon
    # is_beacon[0,1]  = True   # seg=1 内 beacon
    # is_beacon[0,2]  = True   # seg=1 内 beacon
    # is_beacon[0,3]  = True   # seg=1 内 beacon
    # is_beacon[0,4]  = True   # seg=1 内 beacon
    # is_beacon[0,5]  = True   # seg=1 内 beacon
    # is_beacon[0,6]  = True   # seg=1 内 beacon
    # is_beacon[0,7]  = True   # seg=1 内 beacon
    # is_beacon[0,8]  = True   # seg=1 内 beacon
    # is_beacon[0,9]  = True   # seg=1 内 beacon
    # is_beacon[0,5]  = True   # seg=1 内 beacon
    # is_beacon[0,10] = True   # seg=2 内 beacon
    is_beacon[0,15] = True   # seg=3 内 beacon
    is_beacon[0,65] = True   
    # seg==0（global super）天然是 super 纵列

    # --- 被测：混合稀疏 prefill ---
    beacon = BeaconMixedSparsePrefill(
        block_size_M=32, block_size_N=32,  # Triton kernel 友好
    )
    # beacon.set_meta_data(segment_ids, is_beacon)
    out_sparse = beacon(q, k, v, segment_ids, is_beacon)   # [B,Hq,T,Dh]
    print(out_sparse)

    # --- 参考：dense + 等价 mask ---
    # 注意：dense 参考也需要把 KV 头展开到 Hq（为了对齐 H 维度）
    if Hq % Hkv != 0:
        raise RuntimeError("Hq 必须是 Hkv 的整数倍以对齐 GQA。")
    rep = Hq // Hkv
    k_full = k.repeat_interleave(rep, dim=1)
    v_full = v.repeat_interleave(rep, dim=1)
    out_dense = dense_reference_attention(q, k_full, v_full, segment_ids, is_beacon, diag_window=diag_window)
    print(out_dense)

    # --- 一致性断言 ---
    atol = 5e-3 if q.dtype == torch.bfloat16 else 1e-5
    rtol = 5e-3 if q.dtype == torch.bfloat16 else 1e-5
    diff = (out_sparse - out_dense).abs().max().item()
    print(f"max |sparse - dense| = {diff:.6f}")
    assert torch.allclose(out_sparse, out_dense, atol=atol, rtol=rtol), \
        f"Mismatch! max diff {diff}"

    # --- 额外：验证“竖直列”可见性（beacon/super 对所有后续 i 可见） ---
    # 选择一个 beacon 位置 j_b=10，检查 i>j_b 的行里，该列不为 -inf（即参与了注意力）
    j_b = 10
    # 用 dense 构造的 bias 看可见性（true=可见）
    B1, H1, T1, Dh1 = q.shape
    seg = segment_ids
    bec = is_beacon
    super_mask = (seg == 0) | bec
    visible_any = []
    for i in range(j_b+1, T1):
        # i 行是否允许看到 j_b？
        si = int(seg[0, i].item())
        seg_start_i = 0 if si == 0 else int((seg[0][:i+1] == si).nonzero()[:,0].min().item())
        left_bound = seg_start_i if diag_window is None else max(seg_start_i, i - diag_window + 1)
        in_seg = (seg[0, j_b] == si) and (j_b >= left_bound) and (j_b <= i)
        sup_ok = bool(super_mask[0, j_b] and (j_b <= i))
        visible_any.append(bool(in_seg or sup_ok))
    print("vertical visibility from beacon j=10 to later i:", all(visible_any))

    print("✅ BeaconMixedSparsePrefill matches dense reference under the intended mask & window.")

if __name__ == "__main__":
    run_demo()