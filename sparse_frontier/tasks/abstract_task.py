import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from sparse_frontier.modelling.tokenizer import Tokenizer


class AbstractTask(ABC):
    """Base class for defining evaluation tasks.
    
    Handles common task functionality like sample generation and evaluation.
    Subclasses must implement check_inputs(), generate_samples() and evaluate().
    """

    def __init__(
        self,
        num_samples: int,
        max_input_tokens: int,
        max_output_tokens: int,
        tokenizer: Tokenizer,
        random_seed: int,
        template_tokens: int = 64,
        **kwargs
    ) -> None:
        """Initialize task parameters.

        Args:
            num_samples: Number of samples to generate
            max_input_tokens: Maximum input sequence length in tokens
            max_output_tokens: Maximum output sequence length in tokens
            tokenizer: Tokenizer for text encoding/decoding
            random_seed: Random seed for reproducibility
            template_tokens: Approximate number of tokens for model's template
        """
        self.num_samples = num_samples
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.tokenizer = tokenizer
        self.random_seed = random_seed
        self.template_tokens = template_tokens
        self.task_params = {}
        self.random_obj = random.Random(self.random_seed)

    def __getattr__(self, name: str) -> Any:
        """Get task parameter by name."""
        if name in self.params:
            return self.params[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def check_sample_length(self, input_text: str, gold_answer: str | list[str]) -> None:
        """Validate input and output sequence lengths.

        Args:
            input_text: Model input text
            gold_answer: Expected model output, either a single string or list of possible answers

        Raises:
            AssertionError: If sequences exceed maximum allowed lengths
        """
        # Handle both single string and list of strings
        if isinstance(gold_answer, str):
            gold_answers = [gold_answer]
        else:
            gold_answers = gold_answer

        # Check length of each possible answer
        for answer in gold_answers:
            gold_length = len(self.tokenizer.text_to_tokens(answer))
            if gold_length > self.max_output_tokens:
                raise AssertionError(
                    f"Gold answer too long: {gold_length} tokens, max {self.max_output_tokens}"
                )
        
        input_length = len(self.tokenizer.text_to_tokens(input_text))
        if input_length > self.max_input_tokens - self.template_tokens:
            raise AssertionError(
                f"Only input too long: {input_length} tokens, max {self.max_input_tokens - self.template_tokens} tokens."
            )

        final_input = self.tokenizer.encode_for_generation(input_text, return_tensors=False)
        final_input_length = len(final_input["input_ids"])

        if final_input_length > self.max_input_tokens:
            raise AssertionError(
                f"Input + Template too long: {final_input_length} tokens, max {self.max_input_tokens} tokens."
            )

        if final_input_length < 0.90 * self.max_input_tokens:
            raise AssertionError(
                f"Input + Template too short: {final_input_length} tokens, min {int(0.90 * self.max_input_tokens)} tokens."
            )

    @abstractmethod
    def check_params(self) -> None:
        """Validate task-specific parameters.

        Raises:
            ValueError: If parameters are missing or invalid
            AssertionError: If parameters fail validation checks
        """
        pass

    @property
    @abstractmethod
    def sample_class(self):
        """Return the sample class to use for this task.
        
        Returns:
            A class that inherits from AbstractSample
        """
        pass

    def generate_samples(self) -> List[Dict[str, Any]]:
        """Generate task evaluation samples.

        Returns:
            List of sample dicts containing:
                input_text: Model input text
                gold_answer: Expected model output
                index: Sample index
                length: Input sequence length
                + Additional task-specific fields
        """
        samples = []
        for i in range(self.num_samples):
            sample = self.sample_class(
                sample_id=i,
                random_seed=self.random_seed,
                max_tokens=self.max_input_tokens - self.template_tokens,
                tokenizer=self.tokenizer,
                task_params=self.task_params,
            )
            self.check_sample_length(sample.input_text, sample.gold_answer)
            samples.append(sample.to_dict())

        return samples

    @staticmethod
    @abstractmethod
    def evaluate(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate model predictions against gold answers.

        Args:
            predictions: List of sample dicts with model predictions in 'pred' field

        Returns:
            Dict of evaluation metrics
        """
        pass
