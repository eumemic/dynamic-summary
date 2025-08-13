"""Relative character count targeting strategy."""

from ..base_strategy import TargetingStrategy


class RelativeCharStrategy(TargetingStrategy):
    """Target a reduction in characters from the original."""
    
    def __init__(self):
        super().__init__("relative_char")
    
    def get_length_instruction(self, input_metrics, target_tokens):
        # Estimate target characters from target tokens
        target_chars = target_tokens * 5
        reduction = input_metrics["characters"] - target_chars
        
        if reduction > 0:
            return f"reducing it by PRECISELY {reduction} characters"
        else:
            # Edge case: target is larger than input
            return f"in PRECISELY {target_chars} characters"
    
    def get_prompt(self, text, input_metrics, target_tokens):
        instruction = self.get_length_instruction(input_metrics, target_tokens)
        return self.get_base_prompt(text, instruction)