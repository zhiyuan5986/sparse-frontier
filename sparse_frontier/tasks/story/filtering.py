"""Task for evaluating model's ability to identify chapters without purchases.

This module implements a task where models need to identify chapters in a narrative where
the protagonist did not make any purchases. The task tests the model's ability to:
1. Follow precise instructions
2. Track item transactions across a narrative
3. Format responses according to a specified template
"""

import re
from typing import Any, Dict, List, Set, Tuple
import copy
import numpy as np

from sparse_frontier.tasks.abstract_task import AbstractTask
from sparse_frontier.tasks.abstract_sample import AbstractSample
from sparse_frontier.tasks.story.narrative import Chapter, NarrativeGenerator
from sparse_frontier.tasks.story.templates import TASK_INTRO
from sparse_frontier.tasks.abstract_prompt import SINGLEQ_PROMPT_TEMPLATE as PROMPT_TEMPLATE

QUESTION = """Identify all chapters where the protagonist did not buy any item.
Note: There are exactly {num_chapters} chapters without any purchases."""

ANSWER_FORMAT = "chapter_id_1, chapter_id_2, ..."

EXTRA_INSTRUCTIONS = """
- In the answer section, provide only the chapter IDs separated by commas.
""".strip()

PROMPT = PROMPT_TEMPLATE.format(
    task_intro=TASK_INTRO,
    question="{question}",
    context="{context}",
    answer_format=ANSWER_FORMAT,
    extra_instructions=EXTRA_INSTRUCTIONS
)


class FilteringSample(AbstractSample):
    """Represents a single filtering task sample."""

    @staticmethod
    def _remove_buying_transactions(chapter: Chapter, items_to_remove: Set[str]) -> None:
        """Remove buying transactions for specified items from a chapter."""
        new_bought = []
        new_buying_trans = []
        
        for item, trans in zip(chapter.bought_items, chapter.structure['buying_transactions']):
            if item not in items_to_remove:
                new_bought.append(item)
                new_buying_trans.append(trans)
                
        chapter.bought_items = new_bought
        chapter.structure['buying_transactions'] = new_buying_trans

    def _modify_chapters(self, chapters: List[Chapter], chapter_ids: List[int]) -> List[Chapter]:
        """Modify chapters by removing purchases from specified chapters and updating subsequent chapters."""
        modified_chapters = copy.deepcopy(chapters)
        items_to_track = set()

        # Remove purchases from specified chapters and track items
        for chapter in modified_chapters:
            if chapter.chapter_id in chapter_ids:
                items_to_track.update(chapter.bought_items)
                self._remove_buying_transactions(chapter, set(chapter.bought_items))

        return modified_chapters

    def _generate_sample(self) -> Tuple[str, str, Dict[str, Any]]:
        """Generate the input text, gold answer and extra data for this sample."""
        # Calculate prompt tokens
        prompt_tokens = len(self.tokenizer.text_to_tokens(
            PROMPT.format(
                context="",
                question=QUESTION.format(num_chapters=1),
            )
        ))
        
        # Calculate remaining tokens for narrative
        narrative_tokens = self.max_tokens - prompt_tokens
        
        # Generate narrative with adjusted token limit
        narrative_gen = NarrativeGenerator(
            tokenizer=self.tokenizer,
            sequence_length=narrative_tokens,
            random_obj=self.random_obj,
        )
        
        # Randomly select chapters to remove purchases from
        num_chapters = len(narrative_gen.chapters)
        chapters_to_modify = sorted(
            self.random_obj.sample(range(1, num_chapters + 1), self.task_params['chapters_in_question'])
        )
        
        modified_chapters = self._modify_chapters(narrative_gen.chapters, chapters_to_modify)
        
        # Build prompt
        context = "\n\n".join(ch.compile_text() for ch in modified_chapters)
        input_text = PROMPT.format(
            context=context,
            question=QUESTION.format(num_chapters=self.task_params['chapters_in_question'])
        )

        # Generate expected output
        gold_answer = ", ".join(map(str, chapters_to_modify)) if chapters_to_modify else ""

        return input_text, gold_answer, {}


class FilteringTask(AbstractTask):
    """Task class for evaluating chapter purchase identification capabilities."""

    def __init__(
        self,
        chapters_in_question: int,
        protagonist_name: str = "Arion",
        **kwargs
    ) -> None:
        """Initialize the filtering task.
        
        Args:
            chapters_in_question: Number of chapters to remove purchases from
            protagonist_name: Name of the story protagonist
            **kwargs: Additional arguments passed to AbstractTask
        """
        super().__init__(**kwargs)
        self.task_params['chapters_in_question'] = chapters_in_question
        self.task_params['protagonist_name'] = protagonist_name
        self.check_params()

    def check_params(self) -> None:
        """Validate task parameters."""
        if 'chapters_in_question' not in self.task_params:
            raise ValueError("Missing required parameter 'chapters_in_question'")
        
        if not isinstance(self.task_params['chapters_in_question'], int):
            raise ValueError("Parameter 'chapters_in_question' must be an integer")
            
        if self.task_params['chapters_in_question'] < 1:
            raise ValueError("Parameter 'chapters_in_question' must be at least 1")

    @property
    def sample_class(self):
        return FilteringSample

    @staticmethod
    def evaluate(examples: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate model predictions against gold answers.
        
        Args:
            examples: List of dictionaries containing predictions and gold answers
            
        Returns:
            Dictionary containing IoU (Intersection over Union) metric and its variance
        """
        def normalize_answer(text: str) -> Set[int]:
            """Extract and normalize chapter IDs from answer text."""
            # Remove any non-numeric characters except commas and whitespace
            text = re.sub(r'[^0-9,\s]', '', text.strip())
            # Split on comma or whitespace
            parts = re.split(r'[,\s]+', text)
            # Convert to integers, ignoring any invalid parts
            try:
                return {int(p) for p in parts if p.strip()}
            except ValueError:
                return set()

        def extract_answer(text: str) -> Set[int]:
            """Extract answer from the formatted response."""
            match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
            if not match:
                return set()
            answer_text = match.group(1)
            
            # Handle both formats: "1, 2, 3" and "Chapter 1, Chapter 2, Chapter 3"
            # First try to extract chapter numbers from "Chapter X" format
            chapter_matches = re.findall(r'Chapter\s+(\d+)', answer_text, re.IGNORECASE)
            if chapter_matches:
                return {int(num) for num in chapter_matches}
            
            # If no "Chapter X" format found, fall back to original number parsing
            return normalize_answer(answer_text)

        sample_ious = []

        for ex in examples:
            pred = ex.get("pred", "").strip()
            gold = ex.get("gold_answer", "").strip()
            
            if not pred or not gold:
                continue

            gold_set = normalize_answer(gold)
            pred_set = extract_answer(pred)

            if not gold_set and not pred_set:
                sample_ious.append(1.0)
                continue

            union = gold_set.union(pred_set)
            intersect = gold_set.intersection(pred_set)
            iou = len(intersect) / len(union) if union else 0.0
            sample_ious.append(iou)

        # Calculate mean and variance
        mean_iou = np.mean(sample_ious) if sample_ious else 0.0
        # Use ddof=1 for unbiased estimate of the variance
        variance = np.var(sample_ious, ddof=1) if len(sample_ious) > 1 else 0.0

        return {
            "iou": mean_iou,
            "iou_variance": variance
        }
