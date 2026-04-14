"""
Stub module - Kaggle functionality has been removed.
These functions are no-ops to maintain code compatibility.
"""


def crawl_descriptions(competition: str, local_data_path: str) -> str:
    """Stub: Kaggle crawler removed."""
    raise NotImplementedError(
        "Kaggle functionality has been removed. "
        f"Use a local description file at {{local_data_path}}/{{competition}}/description.md instead."
    )


def download_data(competition: str, settings, enable_create_debug_data: bool = False) -> None:
    """Stub: Kaggle download removed."""
    raise NotImplementedError(
        "Kaggle functionality has been removed. "
        "Please prepare data manually in the local_data_path directory."
    )


def get_metric_direction(competition: str) -> bool:
    """Stub: Kaggle metric direction lookup removed."""
    raise NotImplementedError(
        "Kaggle functionality has been removed. "
        "Please set metric direction in your competition configuration."
    )
