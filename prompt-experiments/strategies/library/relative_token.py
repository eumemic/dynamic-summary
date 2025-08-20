"""Relative token count targeting strategy."""

from ..base_strategy import TargetingStrategy


class RelativeTokenStrategy(TargetingStrategy):
    """Target a reduction in tokens from the original."""

    def __init__(self):
        super().__init__("relative_token")

    def get_length_instruction(self, input_metrics, target_tokens):
        reduction = input_metrics["tokens"] - target_tokens
        if reduction > 0:
            return f"reducing it by PRECISELY {reduction} tokens"
        else:
            # Edge case: target is larger than input (shouldn't happen in practice)
            return f"in PRECISELY {target_tokens} tokens"

    def get_prompt(self, text, input_metrics, target_tokens):
        instruction = self.get_length_instruction(input_metrics, target_tokens)
        return self.get_base_prompt(text, instruction)
