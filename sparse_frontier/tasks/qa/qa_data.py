import json
from pathlib import Path
from typing import List, Dict, Tuple
from abc import ABC, abstractmethod


def get_dataset(dataset_name: str):
    """Get initialized QA dataset based on dataset name.
    
    Args:
        dataset_name: Name of dataset to load ('squad', 'quality', or 'toeflqa')
        
    Returns:
        Initialized QA dataset with data loaded
    """
    # Get path to data directory relative to this file
    current_dir = Path(__file__).parent
    data_dir = current_dir / 'data'

    if dataset_name == "squad":
        data_path = data_dir / 'squad.json'
        return SQuADDataset(str(data_path))
    elif dataset_name == "quality":
        data_path = data_dir / 'quality.jsonl'
        return QualityDataset(str(data_path))
    elif dataset_name == "toeflqa":
        data_path = data_dir / 'toeflqa.jsonl'
        return TOEFLQADataset(str(data_path))


class QADataset(ABC):
    """Abstract base class defining the interface for QA dataset parsers.
    
    All QA dataset parsers should inherit from this class and implement the read_data method.
    
    Output Format:
    - qa_samples: List[Dict] where each dict contains:
        - context_idx: int, index into unique_contexts list
        - question: str, the question text
        - answer: Union[List[str], str], either:
            - List[str] for open-ended QA (e.g. SQuAD)
            - str for multiple choice QA (e.g. Quality, TOEFL)
    - unique_contexts: List[str], list of all unique context passages
    """
    
    def __init__(self, data_path: str):
        """Initialize the QA dataset parser.
        """
        self.qa_samples, self.unique_contexts = self.read_data(data_path)
        self._remove_duplicate_questions()
    
    def _remove_duplicate_questions(self):
        """Remove samples with duplicate question-context pairs."""
        # Create a set to track seen question-context pairs
        seen_pairs = set()
        filtered_samples = []
        
        for sample in self.qa_samples:
            # Create a unique key from question and context
            key = (sample['question'].strip().lower(), sample['context_idx'])
            
            # Only keep samples with unique question-context pairs
            if key not in seen_pairs:
                seen_pairs.add(key)
                filtered_samples.append(sample)
        
        # Update qa_samples with deduplicated list
        self.qa_samples = filtered_samples
    
    @abstractmethod
    def read_data(self, data_path: str) -> Tuple[List[Dict], List[str]]:
        """Read and process a QA dataset file.
        
        Args:
            data_path: Path to dataset file
            
        Returns:
            Tuple containing:
            - List of processed QA samples with context indices
            - List of unique context documents
        """
        pass

    @property
    @abstractmethod
    def is_mcq(self) -> bool:
        """Whether this dataset contains multiple choice questions."""
        pass


class SQuADDataset(QADataset):
    """Parser for SQuAD format datasets."""
    
    def read_data(self, data_path: str) -> Tuple[List[Dict], List[str]]:
        with open(data_path) as f:
            data = json.load(f)
        
        # Extract and deduplicate contexts
        all_contexts = [p['context'] for d in data['data'] for p in d['paragraphs']]
        unique_contexts = sorted(list(set(all_contexts)))
        context_to_idx = {context: idx for idx, context in enumerate(unique_contexts)}
        
        qa_samples = []
        for article in data['data']:
            for paragraph in article['paragraphs']:
                curr_context_idx = context_to_idx[paragraph['context']]
                
                for qa in paragraph['qas']:
                    if not qa['is_impossible']:
                        qa_samples.append({
                            'context_idx': curr_context_idx,
                            'question': qa['question'],
                            'answer': [a['text'] for a in qa['answers']],
                        })
                        
        return qa_samples, unique_contexts

    @property
    def is_mcq(self) -> bool:
        return False


class JSONLinesQADataset(QADataset):
    """Base parser for JSONL format QA datasets (Quality and TOEFL)."""
    
    def read_data(self, data_path: str) -> Tuple[List[Dict], List[str]]:
        with open(data_path) as f:
            data = [json.loads(line) for line in f]
        
        # First pass: collect unique contexts
        unique_contexts = sorted(list({item['context'] for item in data}))
        context_to_idx = {context: idx for idx, context in enumerate(unique_contexts)}
        
        # Second pass: create QA samples
        qa_samples = []
        for item in data:
            context_idx = context_to_idx[item['context']]
            questions = item['questions']
            answers = item['answer']
            
            # Verify questions and answers match in length
            assert len(questions) == len(answers), \
                f"Mismatch in questions ({len(questions)}) and answers ({len(answers)}) length"
            
            # Create individual QA samples
            for q, a in zip(questions, answers):
                # For TOEFL QA, remove "Question: " prefix if present
                if isinstance(self, TOEFLQADataset) and q.startswith("Question: "):
                    q = q[len("Question: "):]
                
                qa_samples.append({
                    'context_idx': context_idx,
                    'question': q,
                    'answer': a,
                })
        
        return qa_samples, unique_contexts


class QualityDataset(JSONLinesQADataset):
    """Parser for Quality format datasets."""
    
    @property
    def is_mcq(self) -> bool:
        return True


class TOEFLQADataset(JSONLinesQADataset):
    """Parser for TOEFL QA format datasets."""
    
    @property
    def is_mcq(self) -> bool:
        return True
