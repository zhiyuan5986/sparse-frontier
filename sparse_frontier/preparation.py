import os

from sparse_frontier.utils import GlobalSettings
from sparse_frontier.utils.data import write_jsonl, get_data_path
from sparse_frontier.utils.general import save_config
from sparse_frontier.tasks.registry import TASK_REGISTRY


def check_args():
    from transformers import AutoTokenizer, AutoConfig
    
    cfg = GlobalSettings.get("cfg")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(cfg.model.path, trust_remote_code=True)

    if tokenizer.model_max_length < cfg.max_input_tokens + cfg.max_output_tokens:
        raise ValueError(f"Model maximum sequence length ({tokenizer.model_max_length}) is less than required length ({cfg.max_input_tokens + cfg.max_output_tokens})")
    
    max_pos_embeddings = getattr(config, "max_position_embeddings", 131072)
    if max_pos_embeddings < cfg.max_input_tokens + cfg.max_output_tokens and 'qwen' not in cfg.model.name:
        raise ValueError(f"Model maximum position embeddings ({max_pos_embeddings}) is less than required length ({cfg.max_input_tokens + cfg.max_output_tokens})")
    
    seq_length = getattr(config, "seq_length", 131072)
    if seq_length < cfg.max_input_tokens + cfg.max_output_tokens:
        raise ValueError(f"Model maximum sequence length ({seq_length}) is less than required length ({cfg.max_input_tokens + cfg.max_output_tokens})")


def get_task_generator():
    from sparse_frontier.modelling.tokenizer import Tokenizer
    cfg = GlobalSettings.get("cfg")
    task_kwargs = {
        'num_samples': cfg.samples,
        'max_input_tokens': cfg.max_input_tokens,
        'max_output_tokens': cfg.max_output_tokens,
        'tokenizer': Tokenizer(cfg.model.path, device='cpu'),
        'random_seed': cfg.random_seed,
        **cfg.task.get('args', {}),
    }
    return TASK_REGISTRY[cfg.task.name](**task_kwargs)


def prepare_task():
    cfg = GlobalSettings.get("cfg")

    # Explicitly enable tokenizers parallelism during preparation
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    check_args()

    data_path = get_data_path()

    print(f"Preparing {cfg.task.name} with {cfg.samples} samples")
    generator = get_task_generator()
    samples = generator.generate_samples()

    write_jsonl(data_path, samples)
    save_config(os.path.dirname(data_path))
    print(f"Saved {cfg.task.name} with {cfg.samples} samples to {data_path}")
    
    # Disable tokenizers parallelism after preparation
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
