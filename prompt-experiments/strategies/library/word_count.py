"""Word count targeting strategy."""

from ..base_strategy import TargetingStrategy


class WordCountStrategy(TargetingStrategy):
    """Target an exact number of words."""
    
    def __init__(self):
        super().__init__("word_count")
    
    def get_length_instruction(self, input_metrics, target_tokens):
        # Estimate words from target tokens
        # Rough heuristic: ~0.75 words per token on average
        target_words = int(target_tokens * 0.75)
        return f"in PRECISELY {target_words} words"
    
    def get_prompt(self, text, input_metrics, target_tokens):
        instruction = self.get_length_instruction(input_metrics, target_tokens)
        return self.get_base_prompt(text, instruction)