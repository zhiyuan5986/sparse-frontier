"""Task for evaluating common word frequency tracking capabilities of language models.

This module implements a task where models need to identify frequently occurring
words in a list containing both common and rare words. The task tests the model's ability to:
1. Track word frequencies in a long list 
2. Identify words that appear more frequently than others
3. Ignore less frequent distractors
"""

from typing import List, Dict, Any, Tuple
import re
import numpy as np

from sparse_frontier.tasks.abstract_task import AbstractTask
from sparse_frontier.tasks.abstract_sample import AbstractSample
from sparse_frontier.tasks.abstract_prompt import SINGLEQ_PROMPT_TEMPLATE

TASK_INTRO = """You will be given a numbered list of words. Your task is to identify the most frequently occurring words. You should solve this task by carefully reading and analyzing the word list. Do not attempt to write code or use programming tools to count frequencies. This is a test of your ability to track word frequencies directly."""

QUESTION_TEMPLATE = """The list contains exactly {num_common} words that appear {common_freq} times each. All other words appear {rare_freq} times each. The order of words in the list is randomized.
Your task is to identify the {num_common} words that appear {common_freq} times each."""

ANSWER_FORMAT = """1. word_one
2. word_two
...
{num_common}. word_{num_common}

Note: List exactly {num_common} words, one per line, numbered from 1 to {num_common}."""


class CommonWordSample(AbstractSample):
    """Handles generation of individual common word frequency tracking samples."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _generate_sample(self) -> Tuple[str, str, Dict[str, Any]]:
        """Generate a single common word frequency tracking sample."""

        words = self.task_params['words'].copy()
        self.random_obj.shuffle(words)

        common_words = words[:self.task_params['num_common_words']]
        rare_words = words[self.task_params['num_common_words']:]
        
        # Create list of common words with repetitions
        common_word_list = common_words * self.task_params['common_word_frequency']

        # Binary search to find maximum number of rare words that fit
        left, right = 0, len(rare_words)
        while left < right:
            mid = (left + right + 1) // 2
            
            # Create test word list with current number of rare words
            test_words = (
                common_word_list +
                rare_words[:mid] * self.task_params['rare_word_frequency']
            )
            test_context = '\n'.join(f"{i+1}. {word}" for i, word in enumerate(test_words))
            
            # Format question with current parameters
            question = QUESTION_TEMPLATE.format(
                num_common=self.task_params['num_common_words'],
                common_freq=self.task_params['common_word_frequency'],
                rare_freq=self.task_params['rare_word_frequency']
            )
            
            # Calculate total tokens with full prompt template
            test_input = SINGLEQ_PROMPT_TEMPLATE.format(
                task_intro=TASK_INTRO,
                context=test_context,
                question=question,
                answer_format=ANSWER_FORMAT.format(
                    num_common=self.task_params['num_common_words']
                ),
                extra_instructions=""
            )
            total_tokens = len(self.tokenizer.text_to_tokens(test_input))
            
            if total_tokens <= self.max_tokens:
                left = mid
            else:
                right = mid - 1
        
        num_rare_words = left
        
        # Ensure we have space for at least one rare word
        assert num_rare_words > 0, "No space for rare words after common words"
        
        # Create final word list with optimal number of rare words
        final_words = (
            common_word_list +
            rare_words[:num_rare_words] * self.task_params['rare_word_frequency']
        )
        self.random_obj.shuffle(final_words)
        
        # Format context and question
        context = '\n'.join(f"{i+1}. {word}" for i, word in enumerate(final_words))
        question = QUESTION_TEMPLATE.format(
            num_common=self.task_params['num_common_words'],
            common_freq=self.task_params['common_word_frequency'],
            rare_freq=self.task_params['rare_word_frequency']
        )
        
        input_text = SINGLEQ_PROMPT_TEMPLATE.format(
            task_intro=TASK_INTRO,
            context=context,
            question=question,
            answer_format=ANSWER_FORMAT.format(
                num_common=self.task_params['num_common_words']
            ),
            extra_instructions=""
        )
        
        gold_answer = '\n'.join(f"{i+1}. {word.lower()}" for i, word in enumerate(common_words))

        extra_data = {
            "num_total_words": len(final_words),
            "common_words": [w.lower() for w in common_words],
            "common_frequency": self.task_params['common_word_frequency'],
            "rare_frequency": self.task_params['rare_word_frequency']
        }
        
        return input_text, gold_answer, extra_data


class CommonWordTask(AbstractTask):
    """Task for evaluating common word frequency tracking capabilities."""
    
    def __init__(
        self,
        common_word_frequency: int = 30,
        rare_word_frequency: int = 3,
        num_common_words: int = 10,
        **kwargs
    ) -> None:
        """Initialize common word frequency tracking task.
        
        Args:
            common_word_frequency: Number of times each common word appears
            rare_word_frequency: Number of times each rare word appears
            num_common_words: Number of common words to identify
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)
        self._create_word_list()
        self.task_params.update({
            'common_word_frequency': common_word_frequency,
            'rare_word_frequency': rare_word_frequency,
            'num_common_words': num_common_words,
            'words': self.words,
        })
        self.check_params()
    
    def _create_word_list(self) -> None:
        from wonderwords import random_word
        nouns = random_word._get_words_from_text_file("nounlist.txt")
        adjs = random_word._get_words_from_text_file("adjectivelist.txt")
        verbs = random_word._get_words_from_text_file("verblist.txt")
        words = nouns + adjs + verbs
        self.words = sorted(list(set(words)))
        self.random_obj.shuffle(self.words)
        self.words = [word for word in self.words if '-' not in word]

    def check_params(self) -> None:
        """Validate task parameters."""
        if not isinstance(self.task_params.get('common_word_frequency'), int):
            raise ValueError("common_word_frequency must be an integer")
        if not isinstance(self.task_params.get('rare_word_frequency'), int):
            raise ValueError("rare_word_frequency must be an integer")
        if not isinstance(self.task_params.get('num_common_words'), int):
            raise ValueError("num_common_words must be an integer")
        
        if self.task_params['common_word_frequency'] <= self.task_params['rare_word_frequency']:
            raise ValueError("common_word_frequency must be greater than rare_word_frequency")
        if self.task_params['num_common_words'] < 1:
            raise ValueError("num_common_words must be at least 1")

    @property
    def sample_class(self):
        return CommonWordSample

    @staticmethod
    def evaluate(predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate model predictions against gold answers.
        
        For each prediction, calculates the intersection over union (IoU) between
        the predicted set of common words and the gold set.
        
        Returns mean IoU across all predictions and IoU variance.
        """
        sample_ious = []
        
        for pred in predictions:
            # Extract answer section
            answer_match = re.search(r'<answer>(.*?)</answer>', pred['pred'], re.DOTALL | re.IGNORECASE)
            if not answer_match:
                sample_ious.append(0.0)
                continue
                
            answer_text = answer_match.group(1).strip()
            
            # Extract words from prediction as a set, normalizing and handling edge cases
            pred_words = set()
            for line in answer_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                # Match numbered lines, handling various formats (1., 1), 1-, etc.)
                match = re.match(r'^\d+[.)\-]?\s*(.+)$', line)
                if match:
                    word = match.group(1).strip().lower()
                    if word:
                        pred_words.add(word)

            gold_words = set(pred['common_words'])
            
            # Calculate intersection over union
            intersection = len(pred_words & gold_words)
            union = len(pred_words | gold_words)
            iou = intersection / union if union > 0 else 0.0
            sample_ious.append(iou)
        
        # Calculate mean and variance
        mean_iou = np.mean(sample_ious) if sample_ious else 0.0
        # Use ddof=1 for unbiased estimate of the variance
        variance = np.var(sample_ious, ddof=1) if len(sample_ious) > 1 else 0.0
        
        return {
            'iou': mean_iou,
            'iou_variance': variance
        }
