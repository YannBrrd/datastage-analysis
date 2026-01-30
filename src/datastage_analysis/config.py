"""
Configuration loader for DataStage Migration Analyzer.

Loads settings from config.yaml with sensible defaults.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULTS = {
    'parser': {
        'max_file_size_mb': 510,
        'max_lines': 0,  # 0 = unlimited
        'max_workers': 4,
    },
    'prediction': {
        'success_baseline': 0.85,
        'effort_factor': 1.0,
        'confidence_adjustment': 0.0,
    },
    'glue': {
        'glue_version': '4.0',
        'default_worker_type': 'G.1X',
        'default_num_workers': 2,
        'enable_bookmarks': True,
        'enable_metrics': True,
    },
    'output': {
        'output_dir': './output',
        'wave_size': 50,
    },
    'logging': {
        'level': 'INFO',
    },
}


class Config:
    """Configuration manager for the migration analyzer."""

    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        """Singleton pattern to ensure single config instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """Load configuration from config.yaml or use defaults."""
        self._config = DEFAULTS.copy()

        # Try to find config.yaml
        config_paths = [
            Path('config.yaml'),
            Path('config.yml'),
            Path(__file__).parent.parent.parent / 'config.yaml',
            Path(__file__).parent.parent.parent / 'config.yml',
            Path.home() / '.datastage-analysis' / 'config.yaml',
        ]

        config_file = None
        for path in config_paths:
            if path.exists():
                config_file = path
                break

        if config_file:
            try:
                import yaml
                with open(config_file, 'r') as f:
                    file_config = yaml.safe_load(f) or {}

                # Deep merge with defaults
                self._config = self._deep_merge(DEFAULTS, file_config)
                logger.info(f"Loaded configuration from {config_file}")

            except ImportError:
                logger.warning("PyYAML not installed, using default configuration")
            except Exception as e:
                logger.warning(f"Failed to load config from {config_file}: {e}")
        else:
            logger.debug("No config.yaml found, using defaults")

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def get(self, *keys, default=None) -> Any:
        """
        Get a configuration value by key path.

        Usage:
            config.get('parser', 'max_lines')
            config.get('glue', 'glue_version', default='4.0')
        """
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    @property
    def parser(self) -> Dict[str, Any]:
        """Get parser configuration."""
        return self._config.get('parser', DEFAULTS['parser'])

    @property
    def prediction(self) -> Dict[str, Any]:
        """Get prediction configuration."""
        return self._config.get('prediction', DEFAULTS['prediction'])

    @property
    def glue(self) -> Dict[str, Any]:
        """Get Glue generation configuration."""
        return self._config.get('glue', DEFAULTS['glue'])

    @property
    def output(self) -> Dict[str, Any]:
        """Get output configuration."""
        return self._config.get('output', DEFAULTS['output'])

    def reload(self):
        """Reload configuration from file."""
        self._load_config()


# Global config instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config
