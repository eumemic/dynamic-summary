"""Base strategy class for length targeting experiments."""

from abc import ABC, abstractmethod


class TargetingStrategy(ABC):
    """Abstract base class for summarization length targeting strategies."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_prompt(self, text: str, input_metrics: dict[str, int], target_tokens: int) -> str:
        """Generate the summarization prompt for this strategy.
        
        Args:
            text: The text to summarize
            input_metrics: Dictionary with 'tokens', 'characters', 'words' counts
            target_tokens: Target token count for the summary
            
        Returns:
            Complete prompt string
        """
        pass

    @abstractmethod
    def get_length_instruction(self, input_metrics: dict[str, int], target_tokens: int) -> str:
        """Generate just the length instruction part of the prompt.
        
        Args:
            input_metrics: Dictionary with 'tokens', 'characters', 'words' counts  
            target_tokens: Target token count for the summary
            
        Returns:
            Length instruction string (e.g., "in PRECISELY 200 tokens")
        """
        pass

    def get_base_prompt(self, text: str, length_instruction: str) -> str:
        """Generate the base prompt template with length instruction.
        
        Args:
            text: The text to summarize
            length_instruction: The length targeting instruction
            
        Returns:
            Complete prompt
        """
        return f"""Summarize the following text {length_instruction}:

{text}

Output ONLY the summary, nothing else."""
