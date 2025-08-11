"""Prompt management system for loading and hydrating templates."""

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PromptManager:
    """Single prompt template manager with fast hydration.

    This class manages a single prompt template, loading it once at construction
    and providing fast hydration with validation.
    """

    def __init__(self, template_path: str | Path):
        """Initialize with a single prompt template.

        Args:
            template_path: Path to the prompt template file

        Raises:
            FileNotFoundError: If template file doesn't exist
            ValueError: If template file can't be read
        """
        self.template_path = Path(template_path)

        # Load template once at construction
        if not self.template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {self.template_path}")

        try:
            self.template = self.template_path.read_text(encoding="utf-8")
        except Exception as e:
            raise ValueError(
                f"Failed to read prompt template from {self.template_path}: {e}"
            )

        # Extract variables once at construction
        # Handles both {var} and {var:format} patterns
        self.required_variables = set(
            re.findall(r"\{(\w+)(?::[^}]*)?\}", self.template)
        )

        logger.debug(
            f"Loaded prompt from {self.template_path} with variables: {self.required_variables}"
        )

    def hydrate(self, variables: dict[str, Any]) -> str:
        """Hydrate the template with provided variables.

        This is optimized to be fast - no file I/O or regex parsing,
        just validation and string formatting.

        Args:
            variables: Dictionary of variable names to values

        Returns:
            The hydrated prompt text

        Raises:
            ValueError: If required variables are missing
        """
        # Fast validation using set operations
        provided = set(variables.keys())
        missing = self.required_variables - provided

        if missing:
            raise ValueError(
                f"Missing required variables for prompt {self.template_path.name}: {missing}"
            )

        # Fast string formatting
        try:
            return self.template.format(**variables)
        except KeyError as e:
            # This shouldn't happen if validation passed, but handle gracefully
            raise ValueError(f"Failed to hydrate prompt: missing variable {e}")
        except Exception as e:
            raise ValueError(f"Failed to hydrate prompt: {e}")
