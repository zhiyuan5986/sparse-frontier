import json
import logging
import os
from pathlib import Path
from typing import List, Union

from sparse_frontier.utils.globals import GlobalSettings


def read_jsonl(manifest: Union[Path, str]) -> List[dict]:
    """Read and parse a JSONL file into a list of dictionaries.

    Args:
        manifest: Path to JSONL file to read

    Returns:
        List of dictionaries parsed from JSONL
    
    Raises:
        json.JSONDecodeError: If JSONL parsing fails
        Exception: If file cannot be read
    """
    try:
        with open(manifest, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse line in manifest file {manifest}: {e}")
        raise
    except Exception as e:
        raise Exception(f"Could not read manifest file {manifest}") from e


def write_jsonl(output_path: Union[Path, str], data: List[dict]) -> None:
    """Write a list of dictionaries to a JSONL file.

    Args:
        output_path: Path to output JSONL file
        data: List of dictionaries to serialize
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def build_task_params_str(task_args, max_input_tokens) -> str:
    """Build descriptive string from task args and max_input_tokens.
    
    Args:
        task_args: List of tuples containing task args
        max_input_tokens: Maximum input tokens
        
    Returns:
        String with parameters in format: param1@value1+param2@value2
    """
    params = []
    for arg_name, arg_value in task_args:
        params.append(f"{arg_name}@{arg_value}")
    params.append(f"max_input_tokens@{max_input_tokens}")
    return "+".join(params)


def build_attn_params_str(attn_args) -> str:
    """Build descriptive string for attention config.
    
    Args:
        attn_args: List of tuples containing attention args
        
    Returns:
        String with parameters in format: param1@value1+param2@value2 or 'default'
    """
    attn_params = []
    for arg_name, arg_value in attn_args:
        attn_params.append(f"{arg_name}@{arg_value}")
    return "+".join(attn_params) if attn_params else "default"


def get_pred_dir() -> str:
    """Get the path to the predictions directory based on config settings.
    The path uses @ to separate parameter names from values and + to join parameters,
    which avoids ambiguities and potential issues with bash interpretation.

    Returns:
        Path string constructed from config parameters including task args
    """
    cfg = GlobalSettings.get('cfg')

    return os.path.join(
        cfg.paths.debug if cfg.debug else cfg.paths.predictions,
        cfg.task.name,
        build_task_params_str(cfg.task.get('args', {}).items(), cfg.max_input_tokens),
        cfg.model.name,
        cfg.attention.name,
        build_attn_params_str(cfg.attention.get('args', {}).items()),
    )


def get_results_dir() -> str:
    """Get the path to the results directory based on config settings.
    The path uses @ to separate parameter names from values and + to join parameters,
    which avoids ambiguities and potential issues with bash interpretation.

    Returns:
        Path string constructed from config parameters including task args
    """
    cfg = GlobalSettings.get('cfg')

    return os.path.join(
        cfg.paths.debug if cfg.debug else cfg.paths.results,
        cfg.task.name,
        build_task_params_str(cfg.task.get('args', {}).items(), cfg.max_input_tokens),
        cfg.model.name,
        cfg.attention.name,
        build_attn_params_str(cfg.attention.get('args', {}).items()),
    )


def get_data_dir() -> str:
    cfg = GlobalSettings.get('cfg')

    return os.path.join(
        cfg.paths.debug if cfg.debug else cfg.paths.data,
        cfg.task.name,
        build_task_params_str(cfg.task.get('args', {}).items(), cfg.max_input_tokens),
        cfg.model.name,
    )


def get_data_path() -> str:
    """Get the path to the task's data file.

    Returns:
        Path to data.jsonl in the task directory
    """
    return os.path.join(get_data_dir(), 'data.jsonl')


def get_pred_path() -> str:
    """Get the path to the task's predictions file.

    Returns:
        Path to pred.jsonl in the task directory
    """
    return os.path.join(get_pred_dir(), 'pred.jsonl')


def get_results_path() -> str:
    """Get the path to the task's evaluation results file.

    Returns:
        Path to evaluation_results.json in the task directory
    """

    cfg = GlobalSettings.get('cfg')
    return os.path.join(get_results_dir(), f'evaluation_results_{cfg.samples}.json')


def load_data_without_predictions() -> List[dict]:
    """Load task data excluding samples that already have predictions.
    Only returns samples with indexes up to cfg.samples.

    Returns:
        List of data samples that haven't been predicted yet, limited by index <= cfg.samples.
    """
    cfg = GlobalSettings.get('cfg')
    data_path = get_data_path()
    pred_path = get_pred_path()

    if os.path.exists(pred_path):
        pred_index = {sample['index'] for sample in read_jsonl(pred_path)}
        data = [sample for sample in read_jsonl(data_path) 
                if sample['index'] not in pred_index and sample['index'] < cfg.samples]
    else:
        data = [sample for sample in read_jsonl(data_path)
                if sample['index'] < cfg.samples]
    
    return data
