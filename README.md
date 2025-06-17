<p align="center">
  <img src="./assets/photo.png" width="100%" alt="logo">
</p>

## TL;DR

**The evaluation framework for training-free sparse attention in LLMs**

This repository serves two main purposes:
1. **Reproducing results** from our paper "[The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs](https://arxiv.org/abs/2504.17768)".
2. **Providing a starting point** for your own sparse attention research and development.

### Key Features

- **Comprehensive evaluation suite**: 9 diverse tasks spanning retrieval, multi-hop reasoning, and information aggregation with rigorous sequence length control and standardized preprocessing pipelines.
- **State-of-the-art sparse attention implementations**: 6 SOTA methods covering all major design paradigms—sparse prefilling (Vertical-Slash, Block-Sparse, FlexPrefill), sparse decoding (Quest), and KV cache compression (SnapKV, Ada-SnapKV)—with optimized CUDA kernels and unified interfaces.
- **Model-agnostic scalability**: Universal sparse attention support across all vLLM-compatible models through centralized attention interception. Native tensor parallelism and intelligent workload scheduling automatically optimize GPU utilization across multi-GPU setups.
- **Research-grade extensibility**: Clean modular architecture with abstract base classes designed for rapid prototyping and integration of novel sparse attention patterns and tasks.

### Getting Started with Sparse Attention

If you're new to sparse attention and want to understand how these patterns work, we recommend starting with our companion repository: [nano-sparse-attention](https://github.com/PiotrNawrot/nano-sparse-attention). It provides clean, educational PyTorch implementations with interactive Jupyter notebooks that you can use to experiment and learn before diving into the optimized implementations in this repository.

## Setup

Follow these steps to set up the environment and prepare for running experiments:

1.  **Create Virtual Environment and Install Dependencies:**
    Set up a dedicated Python environment and install the required packages, including compiling custom CUDA kernels

    ```bash
    # Create a virtual environment using Python 3.11
    python3.11 -m venv .venv

    # Activate the virtual environment
    source .venv/bin/activate

    # Upgrade pip and install essential build/utility tools
    pip install --no-cache-dir --upgrade pip setuptools wheel psutil ninja

    # Install PyTorch
    pip install --no-cache-dir torch==2.5.1

    # Install the sparse_frontier project in editable mode
    pip install --no-cache-dir -e .

    # Compile custom CUDA kernels (for MInference attention)
    # Adjust MAX_JOBS based on your system core count for faster compilation
    MAX_JOBS=8 python compile.py build_ext --inplace --build-lib ./sparse_frontier/modelling/attention/minference
    ```
    For reference, the complete list of dependencies used in our experiments is available in `./assets/pipfreeze.txt`. We tested the codebase on both A100 and H100 GPUs.

2.  **Configure Paths:**
    Modify the default configuration file to specify where data, results, and checkpoints should be stored on your system.

    *   Edit the `paths` section in `configs/default.yaml`.

3.  **Download Model Checkpoints:**
    Obtain the pre-trained model weights you intend to evaluate from Hugging Face Hub. We prefer doing this manually instead of downloading it from HF as this way we have better control of what and where we download things.
    
    *   Ensure the final directory structure for the downloaded checkpoints matches the format expected by the corresponding model configuration file (e.g., as defined in `configs/model/qwen_7b.yaml`). The `model.path` variable in these configs should point to the directory containing the model files.

## Where should I look at if I want to:

### Reproduce your experiments

Experiments are launched using the main script `sparse_frontier.main`. The framework uses [Hydra](https://hydra.cc/) for configuration management. All configurations are stored in YAML files within the `sparse_frontier/configs/` directory, organized into three main categories:

- **`attention/`**: Configurations for different sparse attention mechanisms (dense, quest, snapkv, etc.)
- **`task/`**: Configurations for evaluation tasks (RULER, QA, Story tasks)
- **`model/`**: Configurations for different model architectures (Qwen2.5-7B to 72B)

The execution pipeline typically involves three stages, controlled by the `mode` parameter (defaulting to `all`):
1.  **Preparation (`preparation.py`):** Generates and saves task-specific data based on the selected `task` configuration. Tasks are defined in `sparse_frontier/tasks/` (inheriting from [AbstractTask](./sparse_frontier/tasks/abstract_task.py) and [AbstractSample](./sparse_frontier/tasks/abstract_sample.py)) and registered in [TASK_REGISTRY](sparse_frontier/tasks/registry.py).
2.  **Prediction (`prediction.py`):** Runs the specified `model` with the chosen `attention` mechanism on the prepared data, saving the model outputs. Attention mechanisms are implemented in `sparse_frontier/modelling/attention/` and registered in [ATTENTION_REGISTRY](sparse_frontier/modelling/attention/registry.py).
3.  **Evaluation (`evaluation.py`):** Compares the predictions against the gold answers using the task's specific evaluation logic and saves the final metrics.

**Quick Start Examples:**
```bash
# Basic experiment with command line overrides
python -m sparse_frontier.main task=ruler_niah model=qwen_7b attention=quest samples=50

# Override attention parameters
python -m sparse_frontier.main attention=quest attention.args.token_budget=2048
```

For detailed configuration documentation see **[CONFIGURATION.md](CONFIGURATION.md)**.

**Note**: The current framework implementation supports only batch size = 1. This limitation stems from our initial experiments with methods that had kernels supporting only BS=1. Since then, we have followed a simple heuristic: for a given (Model Size, Method, Sequence Length) combination, we find the minimum tensor parallelism (TP) that provides sufficient total GPU memory to handle the evaluation, then use our intra-node scheduler to distribute BS=1 evaluations across the node's GPUs. For the majority of our evaluations, we achieved >95% GPU utilization based on `nvidia-smi`. Nevertheless, higher throughput and GPU utilization could likely be achieved with larger TP and BS>1. We plan to support batch size > 1 in the next release.

### ( Test existing / Develop my own ) Training Free Sparse Attention

We integrate custom sparse attention mechanisms by intercepting and modifying vLLM's standard attention execution flow. Here's a breakdown of the key components involved:

1.  **Patching vLLM's Attention:** We replace vLLM's default `FlashAttentionImpl.forward` method with our custom function, `vllm_patched_forward` (defined in `sparse_frontier/modelling/models/vllm_model.py`). This function serves as the entry point for our custom attention logic within the vLLM generation loop.

2.  **Centralized Handling:** The `vllm_patched_forward` function delegates the core processing to an `AttentionHandler` instance (from `sparse_frontier/modelling/attention/handler.py`). This handler manages layer-specific state (like token counts per head) and differentiates between the prefill and decoding phases of generation.

3.  **Abstract Attention Interface:** The actual attention computation logic for different patterns is encapsulated in classes that inherit from `AbstractAttention` (defined in `sparse_frontier/modelling/attention/abstract_attention.py`). The `AttentionHandler` retrieves the currently configured attention implementation using `get_attention()` (from `sparse_frontier/modelling/attention/registry.py`).

4.  **Implementing a Custom Pattern:** To introduce a new sparse attention mechanism:
    *   Create a new class inheriting from `AbstractAttention`.
    *   Implement the necessary methods based on your pattern's requirements:
        *   `__call__(self, queries, keys, values, layer_idx)`: Implement the attention computation logic for the prefill phase. The default implementation uses standard FlashAttention.
        *   `decode(self, query, keys, values, k_cache, v_cache, cache_seqlens, output, layer_idx)`: Implement the attention computation for the single-token decoding phase, typically involving interaction with the KV cache. The default uses `flash_attn_with_kvcache`. Specific methods like Quest (`efficient_decoding.py`) implement custom logic (e.g., page selection).
        *   `kv_compress(self, queries, keys, values)`: Implement logic to compress or select keys and values *after* the prefill computation, before they are written to the KV cache by `update_kv_cache` in `handler.py`. See `SnapKVCompression` (`kv_compression.py`) for an example.

5.  **Registration and Configuration:** Add your new class to the `ATTENTION_REGISTRY` in `sparse_frontier/modelling/attention/registry.py`. Additionally, create a corresponding YAML configuration file in `sparse_frontier/configs/attention/` that specifies any initialization arguments under `args`. This allows selecting your custom attention mechanism through the experiment configuration files.

### Extend the Test Data

Experimental data generation is handled by task-specific modules located in `sparse_frontier/tasks/`. Each task implements `AbstractTask` and `AbstractSample` subclasses (defined in `sparse_frontier/tasks/abstract_*.py`) to define input / output creation, and task-specific evaluation. Tasks are registered in `sparse_frontier/tasks/registry.py` and selected via configuration (e.g., `task=your_task_name`). The `preparation.py` script orchestrates the generation process based on the configuration, saving the formatted samples. See existing tasks like `QATask` (`qa_task.py`) or the Ruler tasks (`niah.py`, `cwe.py`, `vt.py`) for implementation examples.

## References

### Sparse Attention Patterns

In this repository, we evaluate 6 sparse attention patterns:

| Pattern | Source |
|---------|--------|
| **Vertical-Slash / Block-Sparse** | [Microsoft](https://github.com/microsoft/MInference) |
| **FlexPrefill** | [ByteDance-Seed](https://github.com/ByteDance-Seed/FlexPrefill) |
| **SnapKV** | [FasterDecoding](https://github.com/FasterDecoding/SnapKV) |
| **Ada-SnapKV** | [FFY0](https://github.com/FFY0/AdaKV) |
| **Quest** | [MIT-HAN-Lab](https://github.com/mit-han-lab/Quest) |

We either re-implement these patterns based on the original code or borrow implementations including kernels (for Vertical-Slash and Block-Sparse) from MInference.

### Evaluation Tasks

Our evaluation framework includes the following tasks:

1. **RULER Tasks**: Re-implementation of NIAH, VT, and CWE tasks from [NVIDIA/RULER](https://github.com/NVIDIA/RULER)

2. **QA Tasks**:
   - Toefl and Quality datasets from [LC-VS-RAG](https://github.com/lixinze777/LC_VS_RAG)
   - SQuAD dataset from [NVIDIA/RULER](https://github.com/NVIDIA/RULER)

3. **Novel Story Tasks**: Narrative tasks developed specifically for this project.

## Cite

If you found the repository useful consider citing the paper about this work.

```
@article{nawrot2025sparsefrontier,
      title={The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs}, 
      author={Piotr Nawrot and Robert Li and Renjie Huang and Sebastian Ruder and Kelly Marchisio and Edoardo M. Ponti},
      year={2025},
      journal={arXiv:2504.17768}
      url={https://arxiv.org/abs/2504.17768}, 
}
```

## Issues:

If you have any questions, feel free to raise a Github issue or contact me directly at: piotr.nawrot@ed.ac.uk
