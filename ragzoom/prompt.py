"""Prompt management system for loading and hydrating templates."""

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PromptManager:
    """Manages loading and hydration of prompt templates with validation."""

    def __init__(self, base_path: Path | None = None):
        """Initialize prompt manager.

        Args:
            base_path: Base directory for prompt files. Defaults to prompts/ in package root.
        """
        if base_path is None:
            base_path = Path(__file__).parent.parent / "prompts"
        self.base_path = base_path
        self._cache: dict[str, str] = {}

        # Define allowed directories for custom prompts (security)
        self.allowed_custom_dirs = [
            self.base_path.resolve(),
            Path.cwd().resolve(),  # Current working directory
            (Path.cwd() / "prompts").resolve(),  # ./prompts subdirectory
        ]

    def _validate_custom_path(self, custom_path: str | Path) -> Path:
        """Validate custom path to prevent directory traversal attacks.

        Args:
            custom_path: Path to validate

        Returns:
            Validated Path object

        Raises:
            ValueError: If path is outside allowed directories
        """
        try:
            resolved_path = Path(custom_path).resolve()

            # Check if path is within allowed directories
            is_allowed = any(
                resolved_path.is_relative_to(allowed_dir)
                for allowed_dir in self.allowed_custom_dirs
                if allowed_dir.exists()
            )

            if not is_allowed:
                raise ValueError(
                    f"Custom prompt path '{resolved_path}' is outside allowed directories. "
                    f"Allowed: {', '.join(str(d) for d in self.allowed_custom_dirs)}"
                )

            return resolved_path
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Invalid custom path '{custom_path}': {e}")

    def load_prompt(
        self,
        prompt_name: str,
        custom_path: str | Path | None = None,
        fallback_text: str | None = None,
    ) -> str:
        """Load a prompt template from file.

        Args:
            prompt_name: Name of the prompt (e.g., "summarization/system")
            custom_path: Optional custom path to override default location
            fallback_text: Fallback text if file not found

        Returns:
            The prompt template text

        Raises:
            FileNotFoundError: If prompt file not found and no fallback provided
        """
        # Use custom path if provided, otherwise construct from base path
        if custom_path:
            prompt_path = self._validate_custom_path(custom_path)
        else:
            # Check cache first
            cache_key = str(prompt_name)
            if cache_key in self._cache:
                return self._cache[cache_key]

            prompt_path = self.base_path / f"{prompt_name}.txt"

        try:
            if prompt_path.exists():
                template = prompt_path.read_text(encoding="utf-8")
                logger.debug(f"Loaded prompt '{prompt_name}' from: {prompt_path}")

                # Cache the result if using default path
                if not custom_path:
                    self._cache[str(prompt_name)] = template

                return template
            else:
                if fallback_text is not None:
                    logger.debug(
                        f"Prompt file not found at {prompt_path}, using fallback"
                    )
                    return fallback_text
                else:
                    raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        except Exception as e:
            if fallback_text is not None:
                logger.debug(
                    f"Error loading prompt from {prompt_path}: {e}, using fallback"
                )
                return fallback_text
            else:
                raise

    def hydrate_prompt(
        self, template: str, variables: dict[str, Any], strict: bool = True
    ) -> str:
        """Hydrate a prompt template with variables.

        Uses Python string format syntax: {variable_name}

        Args:
            template: The prompt template containing {variable} placeholders
            variables: Dictionary of variable names to values
            strict: If True, validate that all variables are used and no extras exist

        Returns:
            The hydrated prompt text

        Raises:
            ValueError: If strict=True and validation fails
        """
        # Extract all variable names from the template
        template_vars = set(re.findall(r"\{(\w+)\}", template))
        provided_vars = set(variables.keys())

        if strict:
            # Check for missing variables
            missing = template_vars - provided_vars
            if missing:
                raise ValueError(f"Missing required variables for prompt: {missing}")

            # Note: We don't check for extra variables - templates should be able
            # to ignore variables they don't need, allowing callers to provide
            # a standard set of variables

        # Hydrate the template
        try:
            return template.format(**variables)
        except KeyError as e:
            raise ValueError(f"Failed to hydrate prompt: missing variable {e}")

    def load_and_hydrate(
        self,
        prompt_name: str,
        variables: dict[str, Any] | None = None,
        custom_path: str | Path | None = None,
        fallback_text: str | None = None,
        strict: bool = True,
    ) -> str:
        """Load and hydrate a prompt in one operation.

        Args:
            prompt_name: Name of the prompt (e.g., "summarization/system")
            variables: Variables to hydrate the template with
            custom_path: Optional custom path to override default location
            fallback_text: Fallback text if file not found
            strict: If True, validate variable usage

        Returns:
            The loaded and hydrated prompt text
        """
        template = self.load_prompt(prompt_name, custom_path, fallback_text)

        if variables:
            return self.hydrate_prompt(template, variables, strict)
        else:
            # If no variables provided, check that template has no placeholders
            if strict:
                template_vars = set(re.findall(r"\{(\w+)\}", template))
                if template_vars:
                    raise ValueError(
                        f"Template requires variables but none provided: {template_vars}"
                    )
            return template
