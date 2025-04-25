import random
from abc import ABC, abstractmethod
from typing import Any, Dict

from sparse_frontier.modelling.tokenizer import Tokenizer


class AbstractSample(ABC):
    """Base class for task samples.
    
    Handles common sample functionality like length validation and conversion to dict format.
    Subclasses must implement _generate_sample().
    """

    def __init__(
        self,
        sample_id: int,
        random_seed: int,
        max_tokens: int,
        tokenizer: Tokenizer,
        task_params: Dict[str, Any]
    ) -> None:
        """Initialize sample parameters.

        Args:
            sample_id: Sample index number
            random_seed: Random seed for reproducibility
            max_tokens: Maximum input sequence length in tokens
            tokenizer: Tokenizer for text encoding/decoding
            task_params: Dictionary of task parameters
        """
        self.sample_id = sample_id
        self.max_tokens = max_tokens
        self.random_obj = random.Random(random_seed + sample_id)
        self.tokenizer = tokenizer
        self.task_params = task_params
        self.input_text, self.gold_answer, self.extra_data = self._generate_sample()

    @abstractmethod
    def _generate_sample(self) -> tuple[str, str, Dict[str, Any]]:
        """Generate the input text, gold answer and extra data for this sample.
        
        Returns:
            Tuple containing:
                - Input text string to be provided to the model
                - Expected gold answer string
                - Dictionary of extra data to include in to_dict output
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to dictionary format.

        Returns:
            Dictionary containing sample data and any extra data from _generate_sample
        """
        # Handle both single string and list of strings for gold_answer
        reference_answer = self.gold_answer[0] if isinstance(self.gold_answer, list) else self.gold_answer
        
        total_sample_length = len(self.tokenizer.encode_for_generation(self.input_text, return_tensors=False)['input_ids']) + \
                             len(self.tokenizer.text_to_tokens(reference_answer))

        base_dict = {
            "index": self.sample_id,
            "input_text": self.input_text,
            "gold_answer": self.gold_answer,
            "length": total_sample_length
        }
        base_dict.update(self.extra_data)
        return base_dict
