"""Shorten by percentage strategy - proven to work well."""

from ..base_strategy import TargetingStrategy


class ShortenPercentageStrategy(TargetingStrategy):
    """Target length by asking to 'shorten by X%'."""

    def __init__(self):
        super().__init__("shorten_percentage")

    def get_length_instruction(self, input_metrics, target_tokens):
        # Calculate reduction percentage
        reduction = ((input_metrics["tokens"] - target_tokens) / input_metrics["tokens"]) * 100
        reduction = round(reduction, 1)

        return f"by PRECISELY {reduction}%"

    def get_prompt(self, text, input_metrics, target_tokens):
        instruction = self.get_length_instruction(input_metrics, target_tokens)
        # Use "Shorten" which performed perfectly in our test
        return f"Shorten the following text {instruction}:\n\n{text}"
