import re
import string
from typing import Set


def normalize_mcq_answer(answer: str) -> str:
    """Normalize multiple choice answer to extract just the letter choices.
    
    Args:
        answer: Raw answer string that may contain explanation tags and other text
        
    Returns:
        Normalized string containing only the answer letters (e.g., 'A' or 'AB')
    """
    # First try to extract answer from tags if present
    answer_match = re.search(r'<answer>(.*?)</answer>', answer, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1)

    # Remove any "Question X" prefix
    answer = re.sub(r'Question\s*\d+\.?\s*', '', answer)
    
    # Clean up answer text
    answer = answer.upper()
    answer = re.sub(r'[^A-D]', '', answer)
    
    # Sort multiple choices to ensure consistent ordering (e.g., 'BA' -> 'AB')
    answer = ''.join(sorted(answer))
        
    return answer


def extract_tagged_response(text: str) -> tuple[str, str]:
    """Extract explanation and answer from tagged response.
    
    Args:
        text: Full response text containing tagged sections
        
    Returns:
        Tuple of (explanation text, answer text)
        Returns empty strings if tags are not found
    """
    explanation = ''
    answer = ''
    
    explanation_match = re.search(r'<explanation>(.*?)</explanation>', text, re.DOTALL)
    if explanation_match:
        explanation = explanation_match.group(1).strip()
        
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
        
    return explanation, answer


def normalize_answer(s: str) -> str:
    """Normalize open-ended answer for comparison.
    
    Args:
        s: Raw answer string
        
    Returns:
        Normalized answer with consistent formatting
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the|and|or|about|to)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    # Extract answer from tags if present
    answer_match = re.search(r'<answer>(.*?)</answer>', s, re.DOTALL)
    if answer_match:
        s = answer_match.group(1)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def get_token_overlap(pred: str, gold: str) -> tuple[Set[str], Set[str]]:
    """Get overlapping tokens between prediction and gold answer.
    
    Args:
        pred: Predicted answer text
        gold: Gold answer text
        
    Returns:
        Tuple of (prediction tokens, gold tokens)
    """
    pred_tokens = set(pred.split())
    gold_tokens = set(gold.split())
    return pred_tokens, gold_tokens


def f1_score(prediction: str, gold: str) -> float:
    """Calculate F1 score between prediction and gold answer.
    
    Args:
        prediction: Predicted answer text
        gold: Gold answer text
        
    Returns:
        F1 score between 0 and 1
    """
    prediction_tokens, gold_tokens = get_token_overlap(prediction, gold)
    
    true_positives = len(prediction_tokens & gold_tokens)
    false_positives = len(prediction_tokens - gold_tokens)
    false_negatives = len(gold_tokens - prediction_tokens)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return round(f1, 2)
