"""Tests for loading and validating YAML experiment configs."""

from pathlib import Path

import yaml
import pytest


CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

# Collect all YAML config files
ALL_CONFIGS = sorted(CONFIGS_DIR.rglob("*.yaml"))

REQUIRED_TOP_KEYS = {"experiment", "data", "model", "training"}


@pytest.mark.parametrize(
    "config_path",
    ALL_CONFIGS,
    ids=[str(p.relative_to(CONFIGS_DIR)) for p in ALL_CONFIGS],
)
class TestConfigLoads:
    """Validate that every YAML config is parseable and has required keys."""

    def test_yaml_parses(self, config_path: Path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert isinstance(config, dict), f"{config_path} did not parse as a dict"

    def test_required_keys_present(self, config_path: Path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        missing = REQUIRED_TOP_KEYS - set(config.keys())
        assert not missing, f"{config_path} missing keys: {missing}"

    def test_model_has_architecture(self, config_path: Path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert "architecture" in config["model"], f"{config_path} model needs 'architecture'"

    def test_training_has_epochs(self, config_path: Path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert "epochs" in config["training"], f"{config_path} training needs 'epochs'"
        assert isinstance(config["training"]["epochs"], int)

    def test_training_has_learning_rate(self, config_path: Path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert "learning_rate" in config["training"]
        assert config["training"]["learning_rate"] > 0
