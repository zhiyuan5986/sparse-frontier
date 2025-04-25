"""Task for evaluating variable tracking capabilities of language models.

This module implements a task where models need to track chains of variable assignments
in a document containing both relevant assignments and distractors. The task tests the model's
ability to:
1. Track multiple variable assignments through chains
2. Identify all variables that eventually get assigned a specific value
3. Ignore irrelevant distractors and noise
"""

from typing import List, Dict, Any, Tuple
import string
import re
import numpy as np

from sparse_frontier.tasks.abstract_task import AbstractTask
from sparse_frontier.tasks.abstract_sample import AbstractSample
from sparse_frontier.tasks.abstract_prompt import SINGLEQ_PROMPT_TEMPLATE

NOISE_SENTENCE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."

# Task introduction and instructions
TASK_INTRO = """I will provide you with a text containing variable assignments. The text contains two types of assignments:
1. Numeric assignments that set a variable to a number (e.g., "VAR ABC = 12345")
2. Copy assignments that set a variable equal to another variable (e.g., "VAR XYZ = VAR ABC")
Variables are sequences of uppercase letters. The assignments can appear in any order in the text."""

ANSWER_FORMAT = "VARIABLE_ONE VARIABLE_TWO etc."

EXTRA_INSTRUCTIONS = """
- List ONLY the variable names that resolve to the target value.
- Variables can be listed in any order.
- Do not include "VAR" prefix in your answer. Do not include punctuation.
""".strip()

QUESTION_TEMPLATE = "Which variables resolve to the value {target_value}? A variable resolves to {target_value} if it is either directly assigned {target_value}, or assigned to another variable that resolves to {target_value}."


class VariableTrackingSample(AbstractSample):
    """Handles generation of individual variable tracking samples."""
    
    def _generate_random_var(self) -> str:
        """Generate a random 5-letter uppercase variable name."""
        return ''.join(self.random_obj.choices(string.ascii_uppercase, k=5))

    def _generate_unique_vars(self, num_vars: int) -> List[str]:
        """Generate a list of unique variable names.
        
        Args:
            num_vars: Number of unique variable names needed
            
        Returns:
            List of unique variable names
        """
        unique_vars = []
        while len(unique_vars) < num_vars:
            new_var = self._generate_random_var()
            if new_var not in unique_vars:
                unique_vars.append(new_var)
        return unique_vars

    def _create_chain(self, vars_for_chain: List[str], initial_value: int) -> List[str]:
        """Create a single chain of variable assignments.
        
        Args:
            vars_for_chain: List of variables to use in the chain
            initial_value: Starting numeric value for the chain
            
        Returns:
            List of assignment statements forming the chain
        """
        chain = [f"VAR {vars_for_chain[0]} = {initial_value}"]
        for i in range(len(vars_for_chain) - 1):
            chain.append(f"VAR {vars_for_chain[i+1]} = VAR {vars_for_chain[i]}")
        return chain

    def _generate_chains(self) -> Tuple[List[List[str]], List[str], str]:
        """Generate variable assignment chains.
        
        Returns:
            Tuple containing:
                - List of variable chains
                - List of variables that get the target value
                - Target value that propagates through first chain
        """
        num_chains = self.task_params['num_chains']
        num_hops = self.task_params['num_hops']
        vars_per_chain = num_hops + 1
        total_vars_needed = num_chains * vars_per_chain
        
        # Generate all unique variables needed
        all_vars = self._generate_unique_vars(total_vars_needed)
        
        # Generate unique integers for each chain
        unique_integers = self.random_obj.sample(range(10000, 99999), num_chains)
        
        # Create chains
        chains = []
        target_vars = None
        target_value = str(unique_integers[0])  # Value that propagates through first chain
        
        for i in range(num_chains):
            # Get variables for this chain
            chain_vars = all_vars[i*vars_per_chain:(i+1)*vars_per_chain]
            chain = self._create_chain(chain_vars, unique_integers[i])
            chains.append(chain)
            
            # Store variables from first chain as target
            if i == 0:
                target_vars = chain_vars
                
        return chains, target_vars, target_value

    def _generate_sample(self) -> Tuple[str, str, Dict[str, Any]]:
        """Generate a single variable tracking sample."""
        chains, target_vars, target_value = self._generate_chains()
        
        # Extract all variable assignment statements
        assignment_statements = []
        for chain in chains:
            assignment_statements.extend(chain)
            
        # Calculate tokens used by assignments and prompt
        assignment_text = " ".join(assignment_statements)
        prompt_tokens = len(self.tokenizer.text_to_tokens(
            SINGLEQ_PROMPT_TEMPLATE.format(
                task_intro=TASK_INTRO,
                context=assignment_text,
                question=QUESTION_TEMPLATE.format(target_value=target_value),
                answer_format=ANSWER_FORMAT,
                extra_instructions=EXTRA_INSTRUCTIONS
            )
        ))
        
        # Calculate how many noise sentences we can add
        noise_tokens = len(self.tokenizer.text_to_tokens(NOISE_SENTENCE))
        remaining_tokens = self.max_tokens - prompt_tokens
        num_noise_sentences = remaining_tokens // noise_tokens
        num_noise_sentences = max(num_noise_sentences - 5, 0) # Safety margin
        
        # Create final list of sentences and shuffle
        sentences = assignment_statements + [NOISE_SENTENCE] * num_noise_sentences
        self.random_obj.shuffle(sentences)
        
        # Format context and question
        context = " ".join(sentences)
        question = QUESTION_TEMPLATE.format(target_value=target_value)
        
        # Format input using template
        input_text = SINGLEQ_PROMPT_TEMPLATE.format(
            task_intro=TASK_INTRO,
            context=context,
            question=question,
            answer_format=ANSWER_FORMAT,
            extra_instructions=EXTRA_INSTRUCTIONS
        )
        
        # Format gold answer
        gold_answer = " ".join(target_vars)
        
        extra_data = {
            "target_value": target_value,
            "num_chains": len(chains),
            "num_hops": self.task_params['num_hops'],
            "target_vars": target_vars
        }
        
        return input_text, gold_answer, extra_data


class VariableTrackingTask(AbstractTask):
    """Task for evaluating variable tracking capabilities."""
    
    def __init__(
        self,
        num_chains: int = 1,
        num_hops: int = 4,
        **kwargs
    ) -> None:
        """Initialize variable tracking task.
        
        Args:
            num_chains: Number of variable chains to include
            num_hops: Number of variable assignments in each chain
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)
        self.task_params.update({
            'num_chains': num_chains,
            'num_hops': num_hops
        })
        self.check_params()

    def check_params(self) -> None:
        """Validate task parameters."""
        if not isinstance(self.task_params.get('num_chains'), int):
            raise ValueError("num_chains must be an integer")
        if not isinstance(self.task_params.get('num_hops'), int):
            raise ValueError("num_hops must be an integer")
        if self.task_params['num_chains'] < 1:
            raise ValueError("num_chains must be at least 1")
        if self.task_params['num_hops'] < 1:
            raise ValueError("num_hops must be at least 1")

    @property
    def sample_class(self):
        return VariableTrackingSample

    @staticmethod
    def evaluate(predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate model predictions against gold answers using intersection over union.
        
        For each prediction, calculates IoU between predicted and gold variable sets:
        IoU = |intersection| / |union|
        
        Returns mean IoU across all predictions and variance.
        """
        def normalize_answer(text: str) -> str:
            """Normalize answer text for comparison."""
            # Extract answer from tagged response if present
            answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
            if answer_match:
                text = answer_match.group(1)
                
            # Convert to uppercase and remove extra whitespace
            text = re.sub(r'\s+', ' ', text.upper().strip())
            # Remove any "VAR" prefixes
            text = re.sub(r'VAR\s+', '', text)
            return text

        sample_ious = []
        
        for pred in predictions:
            # Convert predictions and gold answers to sets of variables
            pred_vars = set(normalize_answer(pred['pred']).split())
            gold_vars = set(normalize_answer(pred['gold_answer']).split())
            
            # Calculate intersection over union
            intersection = len(pred_vars & gold_vars)
            union = len(pred_vars | gold_vars)
            
            # Handle edge case where both sets are empty
            iou = 1.0 if union == 0 else intersection / union
            sample_ious.append(iou)
        
        # Calculate mean and variance of IoU scores
        mean_iou = np.mean(sample_ious) if sample_ious else 0.0
        # Use ddof=1 for unbiased estimate of the variance
        variance = np.var(sample_ious, ddof=1) if len(sample_ious) > 1 else 0.0
        
        return {
            'iou': mean_iou,
            'iou_variance': variance
        }
    