"""Cut by percentage strategy."""

from ..base_strategy import TargetingStrategy


class CutPercentageStrategy(TargetingStrategy):
    """Target length by asking to 'cut by X%'."""
    
    def __init__(self):
        super().__init__("cut_percentage")
    
    def get_length_instruction(self, input_metrics, target_tokens):
        # Calculate reduction percentage
        reduction = ((input_metrics["tokens"] - target_tokens) / input_metrics["tokens"]) * 100
        reduction = round(reduction, 1)
        
        return f"by PRECISELY {reduction}%"
    
    def get_prompt(self, text, input_metrics, target_tokens):
        instruction = self.get_length_instruction(input_metrics, target_tokens)
        return f"Cut the following text {instruction}:\n\n{text}"