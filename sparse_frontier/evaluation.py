from typing import List
import json
from sparse_frontier.utils.general import GlobalSettings
from sparse_frontier.utils.data import get_pred_path, get_data_path, read_jsonl, get_results_path
from sparse_frontier.tasks.registry import TASK_REGISTRY


def merge_data_and_predictions(data: List[dict], predictions: List[dict]) -> List[dict]:
    """Merge data samples with model predictions based on index.
    
    Args:
        data: List of dictionaries containing input data and gold answers
        predictions: List of dictionaries containing model predictions and metrics
        
    Returns:
        List of merged dictionaries with all fields
        
    Raises:
        AssertionError: If indexes don't match between data and predictions
    """
    # Create index mappings
    data_by_index = {item['index']: item for item in data}
    pred_by_index = {item['index']: item for item in predictions}
    
    # Verify all indexes match
    data_indexes = set(data_by_index.keys())
    pred_indexes = set(pred_by_index.keys())
    assert data_indexes == pred_indexes, \
        f"Mismatch between data and prediction indexes. Missing from data: {pred_indexes - data_indexes}. " \
        f"Missing from predictions: {data_indexes - pred_indexes}"

    # Merge data and predictions
    merged = []
    for idx in data_indexes:
        sample = data_by_index[idx].copy()
        sample.update(pred_by_index[idx])
        merged.append(sample)
        
    assert len(merged) == len(data) == len(predictions), \
        f"Merged data length {len(merged)} doesn't match input lengths: data={len(data)}, predictions={len(predictions)}"
        
    return merged


def evaluate_task() -> None:
    cfg = GlobalSettings.get('cfg')

    results_file = get_results_path()

    # Load and merge data and predictions
    data = [x for x in read_jsonl(get_data_path()) if x['index'] < cfg.samples]
    predictions = [x for x in read_jsonl(get_pred_path()) if x['index'] < cfg.samples]
    examples = merge_data_and_predictions(data, predictions)
    
    metrics = TASK_REGISTRY[cfg.task.name].evaluate(examples)
    
    # Add total number of samples evaluated
    metrics['total_samples'] = len(examples)
    
    # Calculate average sparsity
    total_sparsity = sum(example['sparsity'] for example in examples)
    metrics['average_attention_sparsity'] = total_sparsity / len(examples)

    # Calculate average and max output token length if available
    if any('output_tokens_len' in example for example in examples):
        total_output_tokens = sum(example['output_tokens_len'] for example in examples if 'output_tokens_len' in example)
        num_examples_with_len = sum(1 for example in examples if 'output_tokens_len' in example)
        metrics['average_output_tokens'] = total_output_tokens / num_examples_with_len
        metrics['max_output_tokens'] = max(example['output_tokens_len'] for example in examples if 'output_tokens_len' in example)

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)

    print(f'Evaluation results for task {cfg.task.name} saved to {results_file}')
    
    if cfg.print_eval_results:
        print(f'Evaluation results for task {cfg.task.name}:')
        print(json.dumps(metrics, indent=2))
