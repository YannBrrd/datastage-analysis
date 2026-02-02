"""
Target Generator Registry

Manages registration and retrieval of target generators.
Uses a plugin-like architecture to support multiple targets.
"""

import logging
from typing import Dict, Type, Optional, List

from .base import BaseTargetGenerator, TargetConfig

logger = logging.getLogger(__name__)


class TargetRegistry:
    """
    Registry for target generators.

    Allows registering new targets and retrieving them by name.
    Supports both built-in and custom targets.
    """

    _instance: Optional['TargetRegistry'] = None
    _targets: Dict[str, Type[BaseTargetGenerator]] = {}
    _initialized: bool = False

    def __new__(cls) -> 'TargetRegistry':
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the registry."""
        if not self._initialized:
            self._targets = {}
            self._initialized = True

    def register(
        self,
        target_name: str,
        generator_class: Type[BaseTargetGenerator],
        override: bool = False
    ) -> None:
        """
        Register a target generator.

        Args:
            target_name: Unique name for the target (e.g., 'glue', 'sql')
            generator_class: The generator class (must inherit from BaseTargetGenerator)
            override: If True, override existing registration

        Raises:
            ValueError: If target already registered and override=False
            TypeError: If generator_class doesn't inherit from BaseTargetGenerator
        """
        if not issubclass(generator_class, BaseTargetGenerator):
            raise TypeError(
                f"Generator class must inherit from BaseTargetGenerator, "
                f"got {generator_class.__name__}"
            )

        if target_name in self._targets and not override:
            raise ValueError(
                f"Target '{target_name}' already registered. "
                f"Use override=True to replace."
            )

        self._targets[target_name] = generator_class
        logger.debug(f"Registered target generator: {target_name} -> {generator_class.__name__}")

    def unregister(self, target_name: str) -> bool:
        """
        Unregister a target generator.

        Args:
            target_name: Name of the target to unregister

        Returns:
            True if unregistered, False if not found
        """
        if target_name in self._targets:
            del self._targets[target_name]
            logger.debug(f"Unregistered target generator: {target_name}")
            return True
        return False

    def get(
        self,
        target_name: str,
        config: Optional[TargetConfig] = None
    ) -> BaseTargetGenerator:
        """
        Get a target generator instance.

        Args:
            target_name: Name of the target
            config: Optional configuration for the generator

        Returns:
            Configured target generator instance

        Raises:
            KeyError: If target not found
        """
        self._ensure_builtins_registered()

        if target_name not in self._targets:
            available = ', '.join(self.list_targets())
            raise KeyError(
                f"Target '{target_name}' not found. "
                f"Available targets: {available}"
            )

        generator_class = self._targets[target_name]
        return generator_class(config=config)

    def list_targets(self) -> List[str]:
        """
        List all registered target names.

        Returns:
            List of target names
        """
        self._ensure_builtins_registered()
        return list(self._targets.keys())

    def get_target_info(self, target_name: str) -> Dict[str, str]:
        """
        Get information about a target.

        Args:
            target_name: Name of the target

        Returns:
            Dict with target information
        """
        self._ensure_builtins_registered()

        if target_name not in self._targets:
            raise KeyError(f"Target '{target_name}' not found")

        generator_class = self._targets[target_name]
        return {
            'name': target_name,
            'display_name': getattr(generator_class, 'TARGET_DISPLAY_NAME', target_name),
            'main_extension': getattr(generator_class, 'MAIN_SCRIPT_EXTENSION', '.txt'),
            'infrastructure_type': getattr(generator_class, 'INFRASTRUCTURE_TYPE', 'generic'),
            'class': generator_class.__name__,
        }

    def _ensure_builtins_registered(self) -> None:
        """Ensure built-in targets are registered."""
        if 'glue' not in self._targets:
            self._register_builtins()

    def _register_builtins(self) -> None:
        """Register built-in target generators."""
        # Import and register built-in targets
        try:
            from .glue import GlueTargetGenerator
            self.register('glue', GlueTargetGenerator)
        except ImportError as e:
            logger.warning(f"Failed to import Glue target: {e}")

        try:
            from .sql import SQLTargetGenerator
            self.register('sql', SQLTargetGenerator)
        except ImportError as e:
            logger.debug(f"SQL target not available: {e}")

    def clear(self) -> None:
        """Clear all registrations (mainly for testing)."""
        self._targets.clear()


# Module-level singleton instance
_registry = TargetRegistry()


def get_target_generator(
    target_name: str,
    config: Optional[TargetConfig] = None
) -> BaseTargetGenerator:
    """
    Get a target generator by name.

    This is the main entry point for getting target generators.

    Args:
        target_name: Name of the target ('glue', 'sql', etc.)
        config: Optional configuration

    Returns:
        Configured target generator instance

    Example:
        >>> generator = get_target_generator('glue')
        >>> result = generator.generate(prediction, structure)
    """
    return _registry.get(target_name, config)


def register_target(
    target_name: str,
    generator_class: Type[BaseTargetGenerator],
    override: bool = False
) -> None:
    """
    Register a custom target generator.

    Args:
        target_name: Unique name for the target
        generator_class: The generator class
        override: If True, override existing registration

    Example:
        >>> from datastage_analysis.generation.targets import register_target
        >>> register_target('custom', MyCustomGenerator)
    """
    _registry.register(target_name, generator_class, override)


def list_available_targets() -> List[str]:
    """
    List all available target names.

    Returns:
        List of target names

    Example:
        >>> from datastage_analysis.generation.targets import list_available_targets
        >>> print(list_available_targets())
        ['glue', 'sql']
    """
    return _registry.list_targets()
