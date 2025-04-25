"""Task for evaluating multi-hop reasoning capabilities in narrative comprehension.

This module implements a task where models need to track item acquisitions across chapters
in a narrative. The task tests the model's ability to:
1. Follow precise instructions
2. Extract relevant item information from context
3. Connect information across multiple chapters
4. Format responses according to a specified template
"""

import re
from typing import List, Dict, Any, Tuple

from sparse_frontier.tasks.abstract_task import AbstractTask
from sparse_frontier.tasks.abstract_sample import AbstractSample
from sparse_frontier.tasks.story.narrative import NarrativeGenerator
from sparse_frontier.tasks.story.templates import TASK_INTRO
from sparse_frontier.tasks.abstract_prompt import SINGLEQ_PROMPT_TEMPLATE as PROMPT_TEMPLATE

QUESTION = "What was the last item that the protagonist acquired before acquiring {target_item}?"

ANSWER_FORMAT = "ITEM_NAME"

EXTRA_INSTRUCTIONS = """
- Provide only the item name in the answer section.
- Do not include articles like 'the' or 'a' in your answer.
- The item name must be exactly as mentioned in the text.
""".strip()

PROMPT = PROMPT_TEMPLATE.format(
    task_intro=TASK_INTRO,
    question="{question}",
    context="{context}",
    answer_format=ANSWER_FORMAT,
    extra_instructions=EXTRA_INSTRUCTIONS
)


class MultiHopSample(AbstractSample):
    """Represents a single multi-hop reasoning sample with narrative and item queries."""

    def _generate_sample(self) -> Tuple[str, str, Dict[str, Any]]:
        """Generate the input text, gold answer and extra data for this sample."""
        # Calculate prompt tokens
        prompt_tokens = len(self.tokenizer.text_to_tokens(
            PROMPT.format(
                context="",
                question=QUESTION.format(target_item="item_component_1 item_component_2 item_component_3"),
            )
        ))
        
        # Calculate remaining tokens for narrative
        narrative_tokens = self.max_tokens - prompt_tokens
        
        # Generate narrative
        narrative_gen = NarrativeGenerator(
            tokenizer=self.tokenizer,
            sequence_length=narrative_tokens,
            random_obj=self.random_obj,
            protagonist_name=self.task_params.get('protagonist_name', "Arion")
        )
        
        # Find chapters where items were bought
        chapters_with_items = [
            ch for ch in narrative_gen.chapters
            if ch.bought_items
        ]
        
        assert len(chapters_with_items) >= 2, "Need at least two chapters with item purchases"
        
        # Select a random chapter with an item purchase (excluding first chapter with items)
        target_chapter = self.random_obj.choice(chapters_with_items[1:])
        target_item = target_chapter.bought_items[0]
        
        # Find the previous chapter where an item was bought
        prev_chapters = [ch for ch in chapters_with_items if ch.chapter_id < target_chapter.chapter_id]
        prev_chapter = prev_chapters[-1]  # Get the most recent previous chapter
        prev_item = prev_chapter.bought_items[0]
        
        # Generate question
        question = QUESTION.format(target_item=target_item)
        
        input_text = PROMPT.format(
            context=narrative_gen.compile_narrative(),
            question=question
        )
        
        return input_text, prev_item, {}


class MultiHopTask(AbstractTask):
    """Main task class for multi-hop reasoning evaluation in narratives."""

    def __init__(
        self,
        protagonist_name: str = "Arion",
        **kwargs
    ) -> None:
        """Initialize the multi-hop task.
        
        Args:
            protagonist_name: Name of the story protagonist
            **kwargs: Additional arguments passed to AbstractTask
        """
        super().__init__(**kwargs)
        self.task_params['protagonist_name'] = protagonist_name
        self.check_params()

    def check_params(self) -> None:
        """Validate task-specific parameters."""
        if not isinstance(self.task_params['protagonist_name'], str):
            raise ValueError("protagonist_name must be a string")

    @property
    def sample_class(self):
        """Return the sample class for this task."""
        return MultiHopSample

    @staticmethod
    def evaluate(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate model predictions against gold answers."""
        def normalize_answer(text: str) -> str:
            """Normalize answer text for comparison."""
            # Convert to lowercase and remove extra whitespace
            text = re.sub(r'\s+', ' ', text.lower().strip())
            # Remove articles
            text = re.sub(r'^(the|a|an)\s+', '', text)
            return text

        def extract_answer(text: str) -> str:
            """Extract answer from the formatted response."""
            # Find content between <answer> tags
            match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
            if not match:
                return ""
            return normalize_answer(match.group(1))

        import numpy as np
        sample_accuracies = []

        for example in examples:
            gold = normalize_answer(example['gold_answer'])
            pred = extract_answer(example['pred'])
            
            if gold and pred:  # Only evaluate if both answers are non-empty
                # Binary accuracy (1 if correct, 0 if incorrect)
                accuracy = 1.0 if pred == gold else 0.0
                sample_accuracies.append(accuracy)

        # Calculate mean and variance
        mean_accuracy = np.mean(sample_accuracies) if sample_accuracies else 0.0
        # Use ddof=1 for unbiased estimate of the variance
        variance = np.var(sample_accuracies, ddof=1) if len(sample_accuracies) > 1 else 0.0

        return {
            'accuracy': mean_accuracy,
            'accuracy_variance': variance
        }
