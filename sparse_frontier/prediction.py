import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any

from tqdm import tqdm

from sparse_frontier.utils.data import load_data_without_predictions, get_pred_path
from sparse_frontier.utils.general import get_free_ports, save_config
from sparse_frontier.utils.globals import GlobalSettings


model = None


def process_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single sample through the model.
    
    Args:
        sample: Dictionary containing input text and metadata
        
    Returns:
        Sample dictionary augmented with model prediction
    """
    global model
    from sparse_frontier.modelling.attention.registry import get_attention

    get_attention().reset_sparsity_statistics()

    output = model.generate(sample['input_text'])

    output_dict = {
        'pred': output['text'],
        'output_tokens_len': output['output_tokens_len'],
        'sparsity': get_attention().calculate_sparsity(),
        'index': sample['index']
    }

    return output_dict


def init_worker() -> None:
    """Initialize worker process with GPU configuration and VLLM model."""
    import torch.multiprocessing as mp
    from sparse_frontier.modelling.models.vllm_model import VLLMModel

    # Explicitly set tokenizers parallelism for each worker
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    cfg = GlobalSettings.get('cfg')
    
    # Check if CUDA_VISIBLE_DEVICES is already set
    preset_gpus = os.environ.get('CUDA_VISIBLE_DEVICES')
    if preset_gpus:
        available_gpus = [int(gpu) for gpu in preset_gpus.split(',')]
        if len(available_gpus) < cfg.gpus:
            raise ValueError(f"Number of available GPUs ({len(available_gpus)}) is less than required ({cfg.gpus})")
    else:
        available_gpus = list(range(cfg.gpus))

    # Set GPU device visibility based on worker index
    if len(mp.current_process()._identity) > 0:
        worker_index = mp.current_process()._identity[0] - 1
        # Calculate GPU slice for this worker from available GPUs
        worker_gpus = available_gpus[worker_index * cfg.tp:(worker_index + 1) * cfg.tp]
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, worker_gpus))
        # Get pre-allocated port for this worker
        worker_port = GlobalSettings.get('worker_ports')[worker_index]
    else:
        # Single worker case - use all available GPUs and first port
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, available_gpus))
        worker_port = GlobalSettings.get('worker_ports')[0]

    # Configure VLLM environment
    os.environ['VLLM_HOST_IP'] = 'localhost'
    os.environ['VLLM_PORT'] = str(worker_port)

    global model
    model = VLLMModel(
        model_path=cfg.model.path,
        max_input_tokens=cfg.max_input_tokens,
        max_output_tokens=cfg.max_output_tokens,
        tensor_parallel_size=cfg.tp,
        seed=cfg.random_seed
    )


def predict_task() -> None:
    cfg = GlobalSettings.get('cfg')

    from sparse_frontier.modelling.attention.registry import configure_attention
    configure_attention()

    data = load_data_without_predictions()

    pred_path = get_pred_path()

    num_workers = cfg.gpus // cfg.tp
    
    # Get free ports for all workers at the start
    free_ports = get_free_ports(num_workers)
    GlobalSettings.set('worker_ports', free_ports)

    if num_workers == 1:
        # Single worker case - process samples sequentially
        init_worker()
        with open(pred_path, 'at', encoding="utf-8", buffering=1) as fout:
            for sample in tqdm(data, total=len(data)):
                sample_results = process_sample(sample)
                fout.write(json.dumps(sample_results) + '\n')
                fout.flush()

        global model
        model = None
    else:
        # Multi-worker case - process samples in parallel
        with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker) as executor, \
            open(pred_path, 'at', encoding="utf-8", buffering=1) as fout:
            futures = {executor.submit(process_sample, sample): sample for sample in data}
            for future in tqdm(as_completed(futures), total=len(data)):
                sample_results = future.result()
                fout.write(json.dumps(sample_results) + '\n')
                fout.flush()

    save_config(os.path.dirname(pred_path))
    print(f'Prediction for task {cfg.task.name} is done. Output is saved to {pred_path}.')
