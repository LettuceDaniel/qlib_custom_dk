from __future__ import annotations

from pydantic_settings import SettingsConfigDict

from rdagent.app.qlib_rd_loop.conf import ModelBasePropSetting


class SotaPropSetting(ModelBasePropSetting):
    model_config = SettingsConfigDict(env_prefix="QLIB_SOTA_", protected_namespaces=())

    sota_csv_path: str = "/workspace/git_ignore_folder/log_all_model_results/potential_sota_model_result.csv"
    """Path to the CSV file containing SOTA model run_ids"""

    knowledge_base_path: str = "/workspace/git_ignore_folder/model_knowledge_base/model_kb_latest.pkl"
    """Path to the SOTA-tagged knowledge base pickle file"""

    based_on_sota_kb: bool = True
    """Whether to use SOTA knowledge base for hypothesis generation"""


SOTA_PROP_SETTING = SotaPropSetting()
