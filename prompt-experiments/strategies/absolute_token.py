"""Absolute token count targeting strategy."""

from .base_strategy import TargetingStrategy


class AbsoluteTokenStrategy(TargetingStrategy):
    """Target an exact number of tokens."""
    
    def __init__(self):
        super().__init__("absolute_token")
    
    def get_length_instruction(self, input_metrics, target_tokens):
        return f"in PRECISELY {target_tokens} tokens"
    
    def get_prompt(self, text, input_metrics, target_tokens):
        instruction = self.get_length_instruction(input_metrics, target_tokens)
        return self.get_base_prompt(text, instruction)