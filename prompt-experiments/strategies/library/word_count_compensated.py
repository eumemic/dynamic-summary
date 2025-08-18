"""Word count strategy with compensation factor."""

from ..base_strategy import TargetingStrategy


class WordCountCompensatedStrategy(TargetingStrategy):
    """Word count with systematic bias compensation."""
    
    def __init__(self, compensation_factor: float = 0.94):
        super().__init__("word_count_compensated")
        self.compensation_factor = compensation_factor
    
    def get_length_instruction(self, input_metrics, target_tokens):
        # Convert tokens to words with standard ratio
        target_words = int(target_tokens * 0.75)
        # Apply compensation for systematic overshoot
        target_words = int(target_words * self.compensation_factor)
        
        return f"in PRECISELY {target_words} words"
    
    def get_prompt(self, text, input_metrics, target_tokens):
        instruction = self.get_length_instruction(input_metrics, target_tokens)
        return self.get_base_prompt(text, instruction)