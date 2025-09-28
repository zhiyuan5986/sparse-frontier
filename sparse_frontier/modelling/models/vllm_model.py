from typing import Dict, Optional, List

import torch
from vllm import LLM, SamplingParams
from vllm.attention.backends.abstract import AttentionType
from vllm.attention.backends.flash_attn import (
    FlashAttentionMetadata,
    get_num_prefill_decode_query_kv_tokens,
)

from sparse_frontier.modelling.attention.registry import get_attention, get_attention_handler
from sparse_frontier.utils.globals import set_vllm_profiling_done, is_vllm_profiling_done
from .abstract_model import AbstractModel
from sparse_frontier.modelling.tokenizer import Tokenizer

# # —— 供 patched forward 读取的全局上下文（简单可靠）——
# _BEACON_SEGMENT_CONTEXT = {
#     "segment_ids": None,  # torch.LongTensor [L]
#     "is_beacon":  None,   # torch.BoolTensor  [L]
# }
# def set_beacon_segment_context(segment_ids: torch.Tensor, is_beacon: torch.Tensor):
#     _BEACON_SEGMENT_CONTEXT["segment_ids"] = segment_ids
#     _BEACON_SEGMENT_CONTEXT["is_beacon"]  = is_beacon

class VLLMModel(AbstractModel):
    def __init__(
        self,
        model_path: str,
        max_input_tokens: int = 8192,
        max_output_tokens: int = 256,
        dtype: torch.dtype = None,
        tensor_parallel_size: int = 1,
        seed: Optional[int] = 43,
    ):
        """
            vLLM uses forking to run the TP. Therefore we can't initialise CUDA,
            before the fork, as it will trigger an error. That's why we don't
            call get_device() etc.
        """
        self.model_path = model_path
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.dtype = dtype or torch.bfloat16
        self.tensor_parallel_size = tensor_parallel_size
        self.seed = seed

        set_vllm_profiling_done(False)

        assert not torch.cuda.is_initialized(), "CUDA is not initialized"
        self.model = self._load_model(self.model_path)
        self.tokenizer = Tokenizer(self.model_path)

    def _load_model(self, model_path: str) -> LLM:
        if 'Qwen' in model_path and self.max_input_tokens + self.max_output_tokens > 32768:
            factor = (self.max_input_tokens + self.max_output_tokens) / 32768
            hf_overrides = {
                "rope_scaling": {
                    "factor": factor,
                    "original_max_position_embeddings": 32768,
                    "rope_type": "yarn"
                }
            }
        else:
            hf_overrides = {}

        model = LLM(
            model=model_path,
            skip_tokenizer_init=True,
            trust_remote_code=True,
            enforce_eager=True,
            seed=self.seed,
            gpu_memory_utilization=0.9,
            max_num_batched_tokens=self.max_input_tokens + self.max_output_tokens,
            max_model_len=self.max_input_tokens + self.max_output_tokens,
            enable_chunked_prefill=False,
            tensor_parallel_size=self.tensor_parallel_size,
            hf_overrides=hf_overrides,
        )

        # Statistics has been accumulated during vLLM profiling
        get_attention().reset_sparsity_statistics()
        set_vllm_profiling_done(True)

        return model

    def _greedy_config(self, max_output_tokens: int) -> Dict:
        return {
            'sampling_params': SamplingParams(
                max_tokens=max_output_tokens,
                temperature=0,
            ),
            'use_tqdm': False,
        }

    @torch.no_grad()
    def generate(
        self,
        input_text: str,
        max_output_tokens: int = None,
    ) -> str:
        max_output_tokens = max_output_tokens or self.max_output_tokens
        model_input = self.tokenizer.encode_for_generation(input_text, return_tensors=False)

        output = self.model.generate(
            prompt_token_ids=model_input['input_ids'],
            **self._greedy_config(max_output_tokens),
        )

        output_ids = output[0].__dict__['outputs'][0].token_ids
        decoded = self.tokenizer.decode(output_ids)

        return {
            'text': decoded[0] if isinstance(decoded, list) else decoded,
            'output_tokens_len': len(output_ids),
        }

class TableBeaconVLLMModel(VLLMModel):
    @torch.no_grad()
    def generate(
        self,
        input_ids: List[int],
        max_output_tokens: Optional[int] = None,
        segment_ids: torch.Tensor = None,   # ★ 新增：与 input_ids 等长
        is_beacon: torch.Tensor = None,    # ★ 新增：与 input_ids 等长
    ) -> Dict[str, object]:
        """
        说明：
        - segment_ids / is_beacon 若提供，长度必须与 tokenizer 得到的 input_ids 对齐；
          否则按全 0 段、全 False beacon 的缺省。
        - 这两份元信息会在调用 vLLM 前注入到 attention/handler 中，供前向/压缩使用。
        """
        max_output_tokens = max_output_tokens or self.max_output_tokens

        # === Tokenize (不初始化 vLLM 内部 tokenizer) ===
        # model_input = self.tokenizer.encode_for_generation(input_text, return_tensors=False)
        T = len(input_ids)

        # === 规范/缺省 segment 与 beacon ===
        if segment_ids is None:
            segment_ids = torch.zeros(T, dtype=torch.long)
        if is_beacon is None:
            is_beacon = torch.zeros(T, dtype=torch.bool)

        assert len(segment_ids) == T, "segment_ids length must match input_ids length"
        assert len(is_beacon) == T, "is_beacon length must match input_ids length"

        # 你的注意力/压缩器实现（继承 AbstractAttention）里请实现 set_token_meta([T], [T])
        attn_algo = get_attention()
        if hasattr(attn_algo, "set_token_meta"):
            attn_algo.set_token_meta(segment_ids, is_beacon)

        handler = get_attention_handler()
        if hasattr(handler, "set_token_meta"):
            handler.set_token_meta(segment_ids, is_beacon)

        # # （可选）为 decode 过程提供默认策略：新 token 继承最后一个 segment，非 beacon
        # if hasattr(handler, "set_decode_defaults"):
        #     last_seg = int(segment_ids[-1]) if T > 0 else 0
        #     handler.set_decode_defaults(default_segment_id=last_seg, default_is_beacon=False)

        # === 执行 vLLM 生成 ===
        output = self.model.generate(
            prompt_token_ids=input_ids,
            **self._greedy_config(max_output_tokens),
        )

        output_ids = output[0].__dict__['outputs'][0].token_ids
        decoded = self.tokenizer.decode(output_ids)

        return {
            'text': decoded[0] if isinstance(decoded, list) else decoded,
            'output_tokens_len': len(output_ids),
        }


def vllm_patched_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: AttentionType = AttentionType.DECODER,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            output: shape = [num_tokens, num_heads, head_size]
            kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        (num_prefill_query_tokens, num_prefill_kv_tokens, _) = \
            get_num_prefill_decode_query_kv_tokens(attn_metadata, attn_type)
        
        if attn_metadata.prefill_metadata:
            get_attention_handler().__call__(
                queries=query[:num_prefill_query_tokens],
                keys=key[:num_prefill_kv_tokens],
                values=value[:num_prefill_kv_tokens],
                kv_cache=kv_cache,
                output=output[:num_prefill_query_tokens],
            )
        else:
            get_attention_handler().__call__(
                queries=query[num_prefill_query_tokens:],
                keys=key[num_prefill_query_tokens:],
                values=value[num_prefill_query_tokens:],
                kv_cache=kv_cache,
                output=output[num_prefill_query_tokens:],
            )

        return output


def swap_vllm_attention():
    from vllm.attention.backends.flash_attn import FlashAttentionImpl
    FlashAttentionImpl.forward = vllm_patched_forward
