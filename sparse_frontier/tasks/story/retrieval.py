"""Task for evaluating story comprehension and information retrieval capabilities.

This module implements a task where models need to extract specific information from
narrative chapters. The task tests the model's ability to:
1. Comprehend multi-chapter narratives
2. Extract relevant information about locations, characters, and items
3. Format responses according to a specified template
4. Provide clear reasoning for its answers
5. Answer questions about specific details from the text
"""

import re
from typing import List, Dict, Any, Tuple

from sparse_frontier.tasks.abstract_task import AbstractTask
from sparse_frontier.tasks.abstract_sample import AbstractSample
from sparse_frontier.tasks.story.narrative import NarrativeGenerator
from sparse_frontier.tasks.story.templates import TASK_INTRO
from sparse_frontier.tasks.abstract_prompt import MULTIPLEQ_PROMPT_TEMPLATE as PROMPT_TEMPLATE

ANSWER_FORMAT = """1. ANSWER_ONE
2. ANSWER_TWO
etc."""

EXTRA_INSTRUCTIONS = """
- For answers, use one line per answer with the number prefix
- Do not include articles like 'the' or 'a' in answers
- Answers should be specific names/items/locations mentioned in the text
""".strip()

PROMPT = PROMPT_TEMPLATE.format(
    task_intro=TASK_INTRO,
    question="{questions}",
    context="{context}",
    answer_format=ANSWER_FORMAT,
    extra_instructions=EXTRA_INSTRUCTIONS
)


class RetrievalSample(AbstractSample):
    SAFETY_TOKENS = 10

    def _generate_sample(self) -> Tuple[str, str, Dict[str, Any]]:
        num_queries = self.task_params['num_queries']
        
        # Calculate tokens needed for prompt and instructions
        prompt_tokens = len(self.tokenizer.text_to_tokens(
            PROMPT.format(context="", questions="")
        ))
        
        # Estimate tokens for questions and explanation
        question_template = "XXX. In Chapter XXX, which specific location/character/item did the protagonist visit/meet/acquire?\n"
        question_tokens = len(self.tokenizer.text_to_tokens(question_template)) * num_queries * 2
        
        # Calculate remaining tokens for narrative
        narrative_tokens = self.max_tokens - (prompt_tokens + question_tokens + self.SAFETY_TOKENS)

        # Generate narrative with adjusted token limit
        narrative_gen = NarrativeGenerator(
            tokenizer=self.tokenizer,
            sequence_length=narrative_tokens,
            random_obj=self.random_obj,
        )

        assert len(narrative_gen.chapters) >= num_queries, "Not enough chapters for the requested complexity."

        # Select random chapters and generate questions
        selected_chapters = self.random_obj.sample(narrative_gen.chapters, num_queries)
        questions_and_answers = []
        
        if num_queries == 3:
            # Fixed query types for 3 questions with clearer phrasing
            query_types = ["location", "character", "item"]
            self.random_obj.shuffle(query_types)
            
            for i, (ch, query_type) in enumerate(zip(selected_chapters, query_types), start=1):
                if query_type == "location":
                    q = f"{i}. In Chapter {ch.chapter_id}, which specific location did the protagonist visit?"
                    a = ch.location
                elif query_type == "character":
                    q = f"{i}. In Chapter {ch.chapter_id}, which character did the protagonist interact with?"
                    a = ch.character
                else:  # item
                    q = f"{i}. In Chapter {ch.chapter_id}, which specific item was acquired by the protagonist?"
                    a = ch.bought_items[0] if ch.bought_items else "None"
                    
                questions_and_answers.append((i, q, a))
        else:
            # Enhanced question generation for other numbers of queries
            for i, ch in enumerate(selected_chapters, start=1):
                query_type = self.random_obj.choice(["location", "character", "item"])
                
                if query_type == "location":
                    q = f"{i}. In Chapter {ch.chapter_id}, which specific location did the protagonist visit?"
                    a = ch.location
                elif query_type == "character":
                    q = f"{i}. In Chapter {ch.chapter_id}, which character did the protagonist interact with?"
                    a = ch.character
                else:
                    q = f"{i}. In Chapter {ch.chapter_id}, which specific item was acquired by the protagonist?"
                    a = ch.bought_items[0] if ch.bought_items else "None"
                    
                questions_and_answers.append((i, q, a))

        input_text = PROMPT.format(
            context=narrative_gen.compile_narrative(),
            questions="\n".join(q for (_, q, _) in questions_and_answers)
        )

        gold_answer = "\n".join(f"{i}. {a}" for (i, _, a) in questions_and_answers)

        return input_text, gold_answer, {}


class RetrievalTask(AbstractTask):
    """Task class for evaluating story comprehension and information retrieval."""

    def __init__(
        self,
        num_queries: int,
        protagonist_name: str = "Arion",
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.task_params['num_queries'] = num_queries
        self.task_params['protagonist_name'] = protagonist_name
        self.check_params()

    def check_params(self) -> None:
        """Validate task-specific parameters."""
        if 'num_queries' not in self.task_params:
            raise ValueError("Missing required parameter 'num_queries'")
        
        if not isinstance(self.task_params['num_queries'], int):
            raise ValueError("Parameter 'num_queries' must be an integer")
            
        if self.task_params['num_queries'] < 1:
            raise AssertionError("Parameter 'num_queries' must be greater than 0")

    @property
    def sample_class(self):
        return RetrievalSample

    @staticmethod
    def evaluate(examples: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate model predictions against gold answers."""
        def normalize_answer(text: str) -> str:
            """Normalize answer text for comparison."""
            # Convert to lowercase and remove extra whitespace
            text = re.sub(r'\s+', ' ', text.lower().strip())
            # Remove articles
            text = re.sub(r'^(the|a|an)\s+', '', text)
            return text

        def extract_answers(text: str) -> Dict[int, str]:
            """Extract answers from text, handling both formats."""
            # First try to find the <answer> section
            answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
            if answer_match:
                text = answer_match.group(1)
            
            # Extract numbered answers
            answers = {}
            for line in text.split('\n'):
                match = re.match(r'^(\d+)[\.:\)]\s*(.+?)\s*$', line.strip())
                if match:
                    idx = int(match.group(1))
                    answer = normalize_answer(match.group(2))
                    if idx not in answers:  # Take first occurrence if duplicates
                        answers[idx] = answer
            return answers

        import numpy as np
        sample_accuracies = []

        for example in examples:
            gold_answers = extract_answers(example['gold_answer'])
            pred_answers = extract_answers(example['pred'])
            
            if not gold_answers:  # Skip examples without gold answers
                continue
                
            correct = 0
            total = len(gold_answers)
            
            for idx, gold in gold_answers.items():
                if idx in pred_answers and pred_answers[idx] == gold:
                    correct += 1
                    
            sample_accuracies.append(correct / total if total > 0 else 0.0)

        # Calculate mean and variance
        mean_accuracy = np.mean(sample_accuracies) if sample_accuracies else 0.0
        # Use ddof=1 for unbiased estimate of the variance
        variance = np.var(sample_accuracies, ddof=1) if len(sample_accuracies) > 1 else 0.0

        return {
            'accuracy': mean_accuracy,
            'accuracy_variance': variance
        }
