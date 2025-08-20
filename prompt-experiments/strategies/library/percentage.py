"""Percentage/compression ratio targeting strategy."""

from ..base_strategy import TargetingStrategy


class PercentageStrategy(TargetingStrategy):
    """Target a percentage of the original length."""

    def __init__(self):
        super().__init__("percentage")

    def get_length_instruction(self, input_metrics, target_tokens):
        # Calculate what percentage the target is of the input
        percentage = (target_tokens / input_metrics["tokens"]) * 100

        # Round to 1 decimal place for cleaner instructions
        percentage = round(percentage, 1)

        return f"to PRECISELY {percentage}% of its original length"

    def get_prompt(self, text, input_metrics, target_tokens):
        instruction = self.get_length_instruction(input_metrics, target_tokens)
        return self.get_base_prompt(text, instruction)
