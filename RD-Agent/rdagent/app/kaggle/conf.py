# Stub module - Kaggle functionality has been removed
# This file exists for backward compatibility with DataScienceBasePropSetting

from pydantic_settings import SettingsConfigDict


class KaggleBasePropSetting:
    """Deprecated: Kaggle functionality has been removed."""
    model_config = SettingsConfigDict(env_prefix="KAGGLE_", protected_namespaces=())
