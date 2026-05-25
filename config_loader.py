"""
Configuration loader and manager for GAN training pipeline.
Supports YAML configuration files with automatic validation and override.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import argparse
import json


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    data_root: str
    image_size: int = 256
    paired_format: str = "separate"
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    shuffle_train: bool = True


@dataclass
class AugmentationConfig:
    """Augmentation configuration."""
    enabled: bool = True
    level: str = "moderate"  # light, moderate, heavy
    normalize: bool = True
    normalization_type: str = "imagenet"


@dataclass
class TrainingConfig:
    """Training configuration."""
    num_epochs: int = 100
    learning_rate: float = 0.0002
    batch_size: int = 16
    device: str = "cuda"


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    generator_type: str = "UNet"
    discriminator_type: str = "PatchGAN"
    in_channels: int = 3
    out_channels: int = 3


class ConfigLoader:
    """Load and manage configurations from YAML files."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config loader.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path) if config_path else None
        self.config = {}

        if self.config_path and self.config_path.exists():
            self._load_yaml()
        else:
            self._load_defaults()

    def _load_yaml(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config from {self.config_path}: {e}")
            self._load_defaults()

    def _load_defaults(self):
        """Load default configuration."""
        self.config = {
            "dataset": {
                "data_root": "./data",
                "image_size": 256,
                "paired_format": "separate",
                "batch_size": 16,
                "num_workers": 4,
            },
            "augmentation": {
                "enabled": True,
                "level": "moderate",
                "normalize": True,
            },
            "training": {
                "num_epochs": 100,
                "learning_rate": 0.0002,
                "device": "cuda",
            },
            "model": {
                "generator_type": "UNet",
                "discriminator_type": "PatchGAN",
            },
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.

        Args:
            key: Configuration key (e.g., "dataset.batch_size")
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """
        Set configuration value by dot-notation key.

        Args:
            key: Configuration key (e.g., "dataset.batch_size")
            value: New value
        """
        keys = key.split(".")
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def get_dataset_config(self) -> DatasetConfig:
        """Get dataset configuration object."""
        dataset_dict = self.config.get("dataset", {})
        return DatasetConfig(**{
            k: dataset_dict.get(k, v)
            for k, v in asdict(DatasetConfig()).items()
        })

    def get_augmentation_config(self) -> AugmentationConfig:
        """Get augmentation configuration object."""
        aug_dict = self.config.get("augmentation", {})
        return AugmentationConfig(**{
            k: aug_dict.get(k, v)
            for k, v in asdict(AugmentationConfig()).items()
        })

    def get_training_config(self) -> TrainingConfig:
        """Get training configuration object."""
        train_dict = self.config.get("training", {})
        return TrainingConfig(**{
            k: train_dict.get(k, v)
            for k, v in asdict(TrainingConfig()).items()
        })

    def get_model_config(self) -> ModelConfig:
        """Get model configuration object."""
        model_dict = self.config.get("model", {})
        return ModelConfig(**{
            k: model_dict.get(k, v)
            for k, v in asdict(ModelConfig()).items()
        })

    def override_from_args(self, args: argparse.Namespace):
        """
        Override configuration from command line arguments.

        Args:
            args: Parsed command line arguments
        """
        for key, value in vars(args).items():
            if value is not None:
                # Convert underscore to dot notation
                config_key = key.replace("_", ".")
                self.set(config_key, value)

    def override_from_dict(self, overrides: Dict[str, Any]):
        """
        Override configuration from dictionary.

        Args:
            overrides: Dictionary of configuration overrides
        """
        for key, value in overrides.items():
            self.set(key, value)

    def save(self, output_path: str):
        """
        Save current configuration to YAML file.

        Args:
            output_path: Path to save configuration
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

        print(f"Configuration saved to {output_path}")

    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return self.config.copy()

    def to_json(self) -> str:
        """Get configuration as JSON string."""
        return json.dumps(self.config, indent=2)

    def print_config(self, section: Optional[str] = None):
        """
        Print configuration to console.

        Args:
            section: Optional section to print (e.g., "dataset")
        """
        if section:
            config_to_print = self.config.get(section, {})
            print(f"\n{section.upper()} Configuration:")
        else:
            config_to_print = self.config
            print("\nFull Configuration:")

        print(json.dumps(config_to_print, indent=2))

    def validate(self) -> bool:
        """
        Validate configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        errors = []

        # Validate dataset
        dataset_cfg = self.config.get("dataset", {})
        if not dataset_cfg.get("data_root"):
            errors.append("dataset.data_root is required")

        data_root = Path(dataset_cfg.get("data_root", ""))
        if data_root != Path("./data") and not data_root.exists():
            errors.append(f"dataset.data_root does not exist: {data_root}")

        # Validate training
        training_cfg = self.config.get("training", {})
        if training_cfg.get("num_epochs", 0) <= 0:
            errors.append("training.num_epochs must be > 0")

        # Validate device
        device = self.config.get("hardware", {}).get("device", "cuda")
        if device not in ["cuda", "cpu"]:
            errors.append(f"Invalid device: {device}")

        if errors:
            print("\nConfiguration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False

        return True


def create_config_parser() -> argparse.ArgumentParser:
    """Create argument parser for configuration overrides."""
    parser = argparse.ArgumentParser(description="GAN Training Configuration")

    # Dataset arguments
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--data_root", type=str, help="Dataset root directory")
    parser.add_argument("--image_size", type=int, default=256, help="Image size")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--num_workers", type=int, help="Number of data workers")

    # Training arguments
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], help="Device to use")

    # Augmentation arguments
    parser.add_argument("--aug_level", type=str, choices=["light", "moderate", "heavy"],
                       help="Augmentation level")
    parser.add_argument("--disable_aug", action="store_true", help="Disable augmentation")

    # Output arguments
    parser.add_argument("--save_config", type=str, help="Save config to this path")
    parser.add_argument("--print_config", action="store_true", help="Print configuration")

    return parser


def setup_config_from_args(args: Optional[argparse.Namespace] = None) -> ConfigLoader:
    """
    Setup configuration from command line arguments.

    Args:
        args: Parsed arguments (if None, parse from sys.argv)

    Returns:
        Configured ConfigLoader instance
    """
    if args is None:
        parser = create_config_parser()
        args = parser.parse_args()

    # Load base configuration
    config = ConfigLoader(config_path=args.config)

    # Override with command line arguments
    overrides = {}
    if args.data_root:
        overrides["dataset.data_root"] = args.data_root
    if args.batch_size:
        overrides["dataset.batch_size"] = args.batch_size
    if args.num_workers:
        overrides["dataset.num_workers"] = args.num_workers
    if args.num_epochs:
        overrides["training.num_epochs"] = args.num_epochs
    if args.learning_rate:
        overrides["training.learning_rate"] = args.learning_rate
    if args.device:
        overrides["hardware.device"] = args.device
    if args.aug_level:
        overrides["augmentation.level"] = args.aug_level
    if args.disable_aug:
        overrides["augmentation.enabled"] = False

    config.override_from_dict(overrides)

    # Validate configuration
    if not config.validate():
        raise ValueError("Configuration validation failed")

    # Print configuration if requested
    if args.print_config:
        config.print_config()

    # Save configuration if requested
    if args.save_config:
        config.save(args.save_config)

    return config


if __name__ == "__main__":
    # Example usage
    print("Configuration Loader Module")
    print("=" * 60)

    # Load default configuration
    config = ConfigLoader()
    config.print_config()

    # Example overrides
    print("\n" + "=" * 60)
    print("Example Override:")
    config.set("dataset.batch_size", 32)
    config.set("training.num_epochs", 200)
    print(f"Batch size: {config.get('dataset.batch_size')}")
    print(f"Num epochs: {config.get('training.num_epochs')}")

    # Validate
    print("\n" + "=" * 60)
    print("Validation Result:", config.validate())
