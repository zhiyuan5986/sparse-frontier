"""Task for evaluating needle-in-a-haystack (NIAH) capabilities of language models.

This module implements a task where models need to extract specific key-value pairs from
a document containing both relevant pairs and distractors. The task tests the model's
ability to:
1. Follow precise instructions
2. Extract relevant information while ignoring distractors
3. Format responses according to a specified template
4. Provide clear reasoning for answers
"""

import string
import re
from typing import List, Dict, Any, Tuple
import numpy as np

from sparse_frontier.tasks.abstract_task import AbstractTask
from sparse_frontier.tasks.abstract_sample import AbstractSample
from sparse_frontier.tasks.abstract_prompt import MULTIPLEQ_PROMPT_TEMPLATE as PROMPT_TEMPLATE

# Task introduction and instructions
TASK_INTRO = """I will provide you with a document containing multiple key-value pairs. Your task is to extract specific values associated with given keys."""

ANSWER_FORMAT = """1. The answer for <key1> is <value1>.
2. The answer for <key2> is <value2>.
etc."""

EXTRA_INSTRUCTIONS = """
- Provide answers in the exact order of the requested keys
- Each answer must follow the format: "<number>. The answer for <key> is <value>."
- Ensure exact key matches - do not modify or paraphrase the keys
- Values must match exactly as they appear in the document
""".strip()


class Needle:
    """Represents a key-value pair in the NIAH task."""
    
    def __init__(self, random_obj, used_keys: set, used_values: set) -> None:
        self.key = self.generate_unique(used_keys, random_obj)
        self.value = self.generate_unique(used_values, random_obj)

    @staticmethod
    def generate_kv_part(random_obj, length: int = 8) -> str:
        return ''.join(random_obj.choices(string.ascii_lowercase + string.digits, k=length))

    @staticmethod
    def generate_kv(random_obj) -> str:
        parts = [Needle.generate_kv_part(random_obj) for _ in range(4)]
        return '-'.join(parts)
    
    @staticmethod
    def generate_unique(used_kv: set, random_obj) -> str:
        while True:
            key = Needle.generate_kv(random_obj)
            if key not in used_kv:
                used_kv.add(key)
                return key

    def to_sentence(self) -> str:
        return f"The value for key {self.key} is: {self.value}."


class NIAHSample(AbstractSample):
    """Represents a single NIAH task sample with queries and distractors."""

    def _generate_sample(self) -> Tuple[str, str, Dict[str, Any]]:
        """Generate the input text, gold answer and extra data for this sample."""
        num_queries = self.task_params['num_queries']
        used_keys = set()
        used_values = set()
        needles = [
            Needle(self.random_obj, used_keys, used_values) 
            for _ in range(num_queries)
        ]

        # Generate query sentences and shuffle
        queries_sentences = [needle.to_sentence() for needle in needles]
        self.random_obj.shuffle(queries_sentences)

        def get_current_token_count(sentences: List[str], keys: List[str]) -> int:
            question = f"Extract the values for the following keys: {', '.join(keys)}"
            text = PROMPT_TEMPLATE.format(
                task_intro=TASK_INTRO,
                context=" ".join(sentences),
                question=question,
                answer_format=ANSWER_FORMAT,
                extra_instructions=EXTRA_INSTRUCTIONS
            )
            return len(self.tokenizer.text_to_tokens(text))

        current_token_count = get_current_token_count(
            queries_sentences, 
            [needle.key for needle in needles]
        )
        tokens_needed = self.max_tokens - current_token_count

        if tokens_needed < 0:
            raise ValueError(
                f"Needles are too long. Current length: {current_token_count}, "
                f"maximum length: {self.max_tokens}"
            )

        # Generate distractor sentences
        distractors_sentences = []
        while True:
            new_distractor = Needle(self.random_obj, used_keys, used_values)
            distractor_sentence = new_distractor.to_sentence()
            distractor_token_length = len(self.tokenizer.text_to_tokens(distractor_sentence))
            if tokens_needed < distractor_token_length:
                break
            distractors_sentences.append(distractor_sentence)
            tokens_needed -= distractor_token_length

        # Combine and shuffle all sentences
        all_sentences = queries_sentences + distractors_sentences
        self.random_obj.shuffle(all_sentences)

        # Generate question
        keys = [needle.key for needle in needles]
        question = f"Extract the values for the following keys: {', '.join(keys)}"

        # Format input using template
        input_text = PROMPT_TEMPLATE.format(
            task_intro=TASK_INTRO,
            context=" ".join(all_sentences),
            question=question,
            answer_format=ANSWER_FORMAT,
            extra_instructions=EXTRA_INSTRUCTIONS
        )

        # Generate gold answer
        gold_answer = "\n".join(
            f"{i+1}. The answer for {needle.key} is {needle.value}." 
            for i, needle in enumerate(needles)
        )

        extra_data = {
            "answers": [(needle.key, needle.value) for needle in needles]
        }

        return input_text, gold_answer, extra_data


class NIAHTask(AbstractTask):
    """Main task class for the Needle-in-a-Haystack evaluation."""

    def __init__(
        self,
        num_queries: int,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.task_params['num_queries'] = num_queries
        self.check_params()

    def check_params(self) -> None:
        if 'num_queries' not in self.task_params:
            raise ValueError("Missing required parameter 'num_queries'")
        
        if not isinstance(self.task_params['num_queries'], int):
            raise ValueError("Parameter 'num_queries' must be an integer")
            
        if self.task_params['num_queries'] < 1:
            raise AssertionError("Parameter 'num_queries' must be greater than or equal to 1")

    @property
    def sample_class(self):
        return NIAHSample

    @staticmethod
    def evaluate(examples: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluates model predictions against gold answers."""
        def normalize_answer(text: str) -> str:
            """Normalize answer text for comparison."""
            # Convert to lowercase and remove extra whitespace
            text = re.sub(r'\s+', ' ', text.lower().strip())
            # Remove optional colon after "is"
            text = re.sub(r'\bis:\s+', 'is ', text)
            return text

        def extract_answers(text: str) -> Dict[int, Tuple[str, str]]:
            """Extract answers from text, handling both formats."""
            # First try to find the <answer> section
            answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
            if answer_match:
                text = answer_match.group(1)
            
            # Extract numbered answers with key-value pairs
            answers = {}
            pattern = re.compile(
                r'(\d+)\.\s*The answer for\s+([\w-]+)\s+is:?\s+(.+?)(?:\.|$)', 
                re.IGNORECASE | re.MULTILINE
            )
            
            for match in pattern.finditer(text):
                idx = int(match.group(1))
                key = match.group(2).strip()
                value = match.group(3).strip()
                if idx not in answers:  # Take first occurrence if duplicates
                    answers[idx] = (key, normalize_answer(value))
            
            return answers

        sample_accuracies = []

        for example in examples:
            answers = example['answers']
            prediction = example['pred']
            
            # Get gold answer key-value pairs
            gold_pairs = {
                i + 1: (key, normalize_answer(value)) 
                for i, (key, value) in enumerate(answers)
            }
            
            # Extract predicted key-value pairs
            pred_pairs = extract_answers(prediction)
            
            # Compare predictions
            correct = 0
            total = len(gold_pairs)
            
            for idx, (gold_key, gold_value) in gold_pairs.items():
                if (idx in pred_pairs and 
                    pred_pairs[idx][0] == gold_key and 
                    pred_pairs[idx][1] == gold_value):
                    correct += 1
            
            # Calculate accuracy for this sample
            sample_accuracies.append(correct / total if total > 0 else 0.0)

        # Calculate mean and variance
        mean_accuracy = np.mean(sample_accuracies) if sample_accuracies else 0.0
        # Use ddof=1 for unbiased estimate of the variance
        variance = np.var(sample_accuracies, ddof=1) if len(sample_accuracies) > 1 else 0.0

        return {
            'accuracy': mean_accuracy,
            'accuracy_variance': variance
        }
