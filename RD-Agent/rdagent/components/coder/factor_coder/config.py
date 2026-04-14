import os
import re
from pathlib import Path
from typing import Optional

from pydantic_settings import SettingsConfigDict

from rdagent.components.coder.CoSTEER.config import CoSTEERSettings, resolve_knowledge_base_paths
from rdagent.utils.env import CondaConf, Env, LocalConf, LocalEnv


class FactorCoSTEERSettings(CoSTEERSettings):
    model_config = SettingsConfigDict(env_prefix="FACTOR_CoSTEER_")

    data_folder: str = "git_ignore_folder/factor_implementation_source_data"
    """Path to the folder containing financial data (default is fundamental data in Qlib)"""

    data_folder_debug: str = "git_ignore_folder/factor_implementation_source_data_debug"
    """Path to the folder containing partial financial data (for debugging)"""

    simple_background: bool = False
    """Whether to use simple background information for code feedback"""

    file_based_execution_timeout: int = 3600
    """Timeout in seconds for each factor implementation execution"""

    select_method: str = "random"
    """Method for the selection of factors implementation"""

    python_bin: str = "python"
    """Path to the Python binary"""

    # Auto-versioning knowledge base settings
    knowledge_base_folder: Optional[str] = None
    """Folder for storing versioned knowledge base files. If set, overrides knowledge_base_path and new_knowledge_base_path for auto-versioning."""

    knowledge_base_name_prefix: str = "factor_kb"
    """Prefix for knowledge base file names (e.g., 'factor_kb' → 'factor_kb_v1.pkl', 'factor_kb_v2.pkl')"""

    auto_knowledge_versioning: bool = True
    """Enable automatic knowledge base versioning (creates new version file on each run)"""


def get_factor_env(
    conf_type: Optional[str] = None,
    extra_volumes: dict = {},
    running_timeout_period: int = 600,
    enable_cache: Optional[bool] = None,
) -> Env:
    conf = FactorCoSTEERSettings()
    # Try to get conda env name from CONDA_DEFAULT_ENV
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")

    if conda_env:
        # Use CondaConf when CONDA_DEFAULT_ENV is set
        env = LocalEnv(conf=(CondaConf(conda_env_name=conda_env)))
    elif conf.python_bin != "python":
        # Derive bin_path from python_bin path
        import re
        python_path = conf.python_bin
        # Extract the bin directory from python_bin path
        bin_path = re.sub(r'/python(\d+\.\d+)?$', '', python_path)
        bin_path = re.sub(r'/python$', '', bin_path)
        # Use LocalConf with custom bin_path
        env = LocalEnv(conf=(LocalConf(default_entry="python main.py", bin_path=bin_path)))
    else:
        # Fall back to LocalConf with default settings
        env = LocalEnv(conf=(LocalConf(default_entry="python main.py")))

    env.conf.extra_volumes = extra_volumes.copy()
    env.conf.running_timeout_period = running_timeout_period
    if enable_cache is not None:
        env.conf.enable_cache = enable_cache
    env.prepare()
    return env


FACTOR_COSTEER_SETTINGS = FactorCoSTEERSettings()
