"""Model information management from models.json.

This module provides a singleton class for accessing model information
(dimensions, costs, capabilities) from the models.json configuration file.
"""

import json
from pathlib import Path
from typing import Optional, TypedDict, cast

from typing_extensions import NotRequired


class EmbeddingModelConfigDict(TypedDict):
    """Type definition for embedding model configuration in models.json."""

    cost_per_1k: float
    dimensions: int


class LLMModelConfigDict(TypedDict):
    """Type definition for LLM model configuration in models.json."""

    input: float
    output: float
    cache_discount: NotRequired[float]
    context_window: int
    reasoning_levels: NotRequired[list[str]]


class ModelsConfigDict(TypedDict, total=False):
    """Type definition for the complete models.json configuration file."""

    embeddings: dict[str, EmbeddingModelConfigDict]
    llms: dict[str, LLMModelConfigDict]
    last_updated: str
    note: str


class ModelInfo:
    """Singleton for accessing model information from models.json.

    This class loads model configuration once and provides efficient
    lookups for model properties like embedding dimensions, costs,
    and capabilities.
    """

    _instance: Optional["ModelInfo"] = None
    _data: ModelsConfigDict = {}

    def __new__(cls) -> "ModelInfo":
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_models()
        return cls._instance

    def _load_models(self) -> None:
        """Load model data from models.json."""
        models_path = Path(__file__).parent / "models.json"

        if not models_path.exists():
            raise FileNotFoundError(
                f"Models configuration file not found at {models_path}"
            )

        try:
            with open(models_path) as f:
                self._data = cast(ModelsConfigDict, json.load(f))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in models file {models_path}: {e}")
        except OSError as e:
            raise OSError(f"Error reading models file {models_path}: {e}")

    def get_embedding_dimensions(self, model: str) -> int:
        """Get the embedding dimensions for a model.

        Args:
            model: The embedding model name

        Returns:
            Number of dimensions for the embedding model

        Raises:
            ValueError: If the model is not found
        """
        if model not in self._data.get("embeddings", {}):
            available = list(self._data.get("embeddings", {}).keys())
            raise ValueError(
                f"Embedding model '{model}' not found. Available models: {available}"
            )

        dimensions = self._data["embeddings"][model].get("dimensions")
        if dimensions is None:
            raise ValueError(f"No dimensions configured for embedding model '{model}'")

        return int(dimensions)

    def get_embedding_cost(self, model: str) -> float:
        """Get the cost per 1K tokens for an embedding model.

        Args:
            model: The embedding model name

        Returns:
            Cost per 1K tokens in USD

        Raises:
            ValueError: If the model is not found
        """
        if model not in self._data.get("embeddings", {}):
            available = list(self._data.get("embeddings", {}).keys())
            raise ValueError(
                f"Embedding model '{model}' not found. Available models: {available}"
            )

        return float(self._data["embeddings"][model]["cost_per_1k"])

    def get_llm_costs(self, model: str) -> tuple[float, float]:
        """Get the input and output costs for an LLM.

        Args:
            model: The LLM model name

        Returns:
            Tuple of (input_cost_per_1k, output_cost_per_1k) in USD

        Raises:
            ValueError: If the model is not found
        """
        if model not in self._data.get("llms", {}):
            available = list(self._data.get("llms", {}).keys())
            raise ValueError(
                f"LLM model '{model}' not found. Available models: {available}"
            )

        llm_info = self._data["llms"][model]
        return float(llm_info["input"]), float(llm_info["output"])

    def get_cache_discount(self, model: str) -> float:
        """Get the cache discount percentage for an LLM.

        Args:
            model: The LLM model name

        Returns:
            Cache discount percentage (e.g., 0.5 = 50% discount, 0.9 = 90% discount)

        Raises:
            ValueError: If the model is not found
        """
        if model not in self._data.get("llms", {}):
            available = list(self._data.get("llms", {}).keys())
            raise ValueError(
                f"LLM model '{model}' not found. Available models: {available}"
            )

        # cache_discount is optional (NotRequired in TypedDict).
        # Default to 0.0 (no discount) for models without caching support.
        return float(self._data["llms"][model].get("cache_discount", 0.0))

    def get_reasoning_levels(self, model: str) -> list[str] | None:
        """Get the supported reasoning effort levels for an LLM.

        Args:
            model: The LLM model name

        Returns:
            List of supported reasoning levels (e.g. ["none", "low", "medium", "high"]),
            or None if the model uses temperature instead of reasoning levels.

        Raises:
            ValueError: If the model is not found
        """
        if model not in self._data.get("llms", {}):
            available = list(self._data.get("llms", {}).keys())
            raise ValueError(
                f"LLM model '{model}' not found. Available models: {available}"
            )

        levels = self._data["llms"][model].get("reasoning_levels")
        if levels is None:
            return None
        return list(levels)

    def supports_temperature(self, model: str) -> bool:
        """Check if an LLM supports temperature parameter.

        Models with reasoning_levels use reasoning effort instead of temperature.

        Args:
            model: The LLM model name

        Returns:
            True if the model supports temperature, False if it uses reasoning levels

        Raises:
            ValueError: If the model is not found
        """
        # Model supports temperature if it doesn't have reasoning_levels
        return self.get_reasoning_levels(model) is None

    def get_context_window(self, model: str) -> int:
        """Get the context window size for an LLM.

        Args:
            model: The LLM model name

        Returns:
            Maximum input tokens for the model

        Raises:
            ValueError: If the model is not found
        """
        if model not in self._data.get("llms", {}):
            available = list(self._data.get("llms", {}).keys())
            raise ValueError(
                f"LLM model '{model}' not found. Available models: {available}"
            )

        return int(self._data["llms"][model]["context_window"])

    def is_gpt5_model(self, model: str) -> bool:
        """Check if a model is a GPT-5 variant.

        Args:
            model: The model name

        Returns:
            True if the model is a GPT-5 variant
        """
        return model.startswith("gpt-5")

    def get_all_embedding_models(self) -> list[str]:
        """Get a list of all available embedding models."""
        return list(self._data.get("embeddings", {}).keys())

    def get_all_llm_models(self) -> list[str]:
        """Get a list of all available LLM models."""
        return list(self._data.get("llms", {}).keys())
