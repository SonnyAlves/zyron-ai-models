import json
import torch
from torch.utils.data import IterableDataset
from typing import Iterator, Dict, Any, List, Optional
import itertools

class ZyronDataset(IterableDataset):
    """
    Production-grade JSONL streaming dataset for Zyron AI.
    Supports efficient streaming of large datasets without loading everything into RAM.
    """
    
    def __init__(
        self, 
        file_path: str, 
        tokenizer: Any, 
        max_seq_len: int = 2048,
        mode: str = "train",
        limit: Optional[int] = None
    ):
        """
        Args:
            file_path: Path to the .jsonl file
            tokenizer: Tokenizer instance (must have encode method)
            max_seq_len: Maximum sequence length
            mode: "train" (infinite stream) or "eval" (one pass)
            limit: If set, only read first N lines (useful for dev mode)
        """
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mode = mode
        self.limit = limit

    def _parse_line(self, line: str) -> Dict[str, Any]:
        """Parse a single JSONL line and tokenize"""
        try:
            data = json.loads(line)
            
            # Handle different data formats (Teacher-Student vs raw)
            if "instruction" in data and "output" in data:
                # Instruction format
                text = f"User: {data['instruction']}\nAssistant: {data['output']}"
            elif "text" in data:
                # Raw text format
                text = data["text"]
            else:
                # Fallback
                text = str(data)
                
            # Tokenize
            # Note: This assumes a HuggingFace-style tokenizer
            # We'll implement a simple fallback if it's a mock tokenizer
            if hasattr(self.tokenizer, "encode"):
                token_ids = self.tokenizer.encode(text)
            else:
                # Fallback for dummy tokenizer in dev mode
                # Simple hash-based tokenization for testing
                token_ids = [hash(w) % 1000 for w in text.split()]

            # Truncate or pad
            if len(token_ids) > self.max_seq_len:
                token_ids = token_ids[:self.max_seq_len]
            else:
                # Pad with 0 (assuming 0 is pad token)
                token_ids = token_ids + [0] * (self.max_seq_len - len(token_ids))
                
            return {
                "input_ids": torch.tensor(token_ids, dtype=torch.long),
                "labels": torch.tensor(token_ids, dtype=torch.long) # Causal LM
            }
            
        except json.JSONDecodeError:
            return None

    def _line_generator(self) -> Iterator[str]:
        """Generator that yields lines from the file"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            if self.limit:
                # Read only first N lines
                for line in itertools.islice(f, self.limit):
                    yield line
            else:
                # Read full file
                for line in f:
                    yield line

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        
        # If in training mode, we want an infinite stream
        while True:
            iterator = self._line_generator()
            
            # Handle multi-worker splitting if needed
            # (Simple implementation: each worker reads full file, 
            # ideal implementation would shard the file)
            
            for line in iterator:
                parsed = self._parse_line(line)
                if parsed is not None:
                    yield parsed
            
            # If eval mode, stop after one pass
            if self.mode != "train":
                break
