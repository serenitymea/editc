import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """Configuration loader for EpicDetector and TrainingDetector.
    
    Loads hyperparameters from a YAML file and provides structured access
    to configuration values.
    """
    
    REQUIRED_SECTIONS = [
        "scoring",
        "windows",
        "optical_flow",
        "video_processing",
        "face_detection",
        "clip_selection",
        "ml"
    ]
    
    def __init__(self, config_path: str):
        """Initialize configuration from YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If required sections are missing
        """
        self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f) or {}
        
        self._validate()
    
    def _validate(self):
        """Validate that all required sections are present in the config."""
        missing_sections = []
        
        for section in self.REQUIRED_SECTIONS:
            if section not in self.config:
                missing_sections.append(section)
        
        if missing_sections:
            raise ValueError(
                f"Missing required configuration sections: {', '.join(missing_sections)}"
            )
    
    def get(self, *keys, default=None) -> Any:
        """Get a configuration value by nested keys.
        
        Args:
            *keys: Sequence of keys to navigate nested config structure
            default: Default value if key path doesn't exist
            
        Returns:
            Configuration value or default
            
        Example:
            config.get("scoring", "heuristic_weights", "motion_p90")
        """
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Dict:
        """Direct dictionary-style access to top-level sections.
        
        Args:
            key: Top-level section name
            
        Returns:
            Configuration section as dictionary
        """
        return self.config[key]


def load_default_config() -> Config:
    """Load configuration from the default config.yaml file.
    
    Returns:
        Config object loaded from config.yaml in current directory
    """
    return Config("config.yaml")