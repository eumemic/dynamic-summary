"""Absolute character count targeting strategy."""

from ..base_strategy import TargetingStrategy


class AbsoluteCharStrategy(TargetingStrategy):
    """Target an exact number of characters."""
    
    def __init__(self):
        super().__init__("absolute_char")
    
    def get_length_instruction(self, input_metrics, target_tokens):
        # Estimate characters from target tokens
        # Rough heuristic: ~5 characters per token on average
        target_chars = target_tokens * 5
        return f"in PRECISELY {target_chars} characters"
    
    def get_prompt(self, text, input_metrics, target_tokens):
        instruction = self.get_length_instruction(input_metrics, target_tokens)
        return self.get_base_prompt(text, instruction)