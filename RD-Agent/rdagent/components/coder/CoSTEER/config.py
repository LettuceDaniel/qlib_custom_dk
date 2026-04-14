import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from rdagent.core.conf import ExtendedBaseSettings

logger = logging.getLogger(__name__)

# Default number of old versions to keep
DEFAULT_KEEP_LAST_N = 5


def cleanup_old_knowledge_base_versions(
    kb_folder: Path,
    prefix: str,
    keep_last_n: int = DEFAULT_KEEP_LAST_N,
    dry_run: bool = False,
) -> list[Path]:
    """
    Clean up old knowledge base version files, keeping only the most recent N versions.

    Args:
        kb_folder: Folder containing knowledge base files
        prefix: Knowledge base prefix (e.g., 'factor_kb', 'model_kb')
        keep_last_n: Number of recent versions to keep (default: 5)
        dry_run: If True, only log what would be deleted without actually deleting

    Returns:
        List of deleted (or would-be-deleted) file paths
    """
    # Match both "prefix_v1.pkl" and "prefix_v1_timestamp.pkl" formats
    pattern = re.compile(r"^" + prefix + r"_v(\d+)(?:_\d{8}_\d{6})?\.pkl$")

    version_files: list[tuple[int, Path]] = []
    for file in kb_folder.glob(f"{prefix}_v*.pkl"):
        # Skip symlinks
        if file.is_symlink():
            continue
        match = pattern.match(file.name)
        if match:
            version_num = int(match.group(1))
            version_files.append((version_num, file))

    # Sort by version number descending (newest first)
    version_files.sort(key=lambda x: x[0], reverse=True)

    # Files to delete = all except the most recent N
    files_to_delete = version_files[keep_last_n:]

    deleted: list[Path] = []
    for version_num, file_path in files_to_delete:
        if dry_run:
            logger.info(f"[DRY-RUN] Would delete old knowledge base version: {file_path} (v{version_num})")
        else:
            try:
                file_path.unlink()
                logger.info(f"Deleted old knowledge base version: {file_path} (v{version_num})")
                deleted.append(file_path)
            except OSError as e:
                logger.warning(f"Failed to delete {file_path}: {e}")

    return deleted


def resolve_knowledge_base_paths(settings) -> tuple[Optional[Path], Optional[Path]]:
    """
    Resolve knowledge base paths with auto-versioning support.

    If auto_versioning is enabled and knowledge_base_folder is set:
    - Finds the latest version file to load as knowledge_base_path
    - Creates a new version file path for new_knowledge_base_path

    Args:
        settings: Settings object with knowledge_base_folder, knowledge_base_name_prefix,
                  auto_knowledge_versioning, knowledge_base_path, new_knowledge_base_path fields

    Returns:
        (knowledge_base_path, new_knowledge_base_path)
    """
    # If auto-versioning is disabled, use original settings
    if not getattr(settings, 'auto_knowledge_versioning', False):
        kb_path = Path(settings.knowledge_base_path) if settings.knowledge_base_path else None
        new_kb_path = Path(settings.new_knowledge_base_path) if settings.new_knowledge_base_path else None
        return kb_path, new_kb_path

    # Auto-versioning mode (even if knowledge_base_folder is None, create a default)
    kb_folder_path = getattr(settings, 'knowledge_base_folder', None)
    if kb_folder_path is None:
        # Create default folder: git_ignore_folder/{prefix}_knowledge_base
        prefix = getattr(settings, 'knowledge_base_name_prefix', 'kb')
        kb_folder_path = f"git_ignore_folder/{prefix}_knowledge_base"
    kb_folder = Path(kb_folder_path)
    kb_folder.mkdir(parents=True, exist_ok=True)

    # Find existing version files
    prefix = getattr(settings, 'knowledge_base_name_prefix', 'kb')
    # Match both "prefix_v1.pkl" and "prefix_v1_timestamp.pkl" formats
    pattern = re.compile(r"^" + prefix + r"_v(\d+)(?:_\d{8}_\d{6})?\.pkl$")
    existing_versions = []

    for file in kb_folder.glob(f"{prefix}_v*.pkl"):
        match = pattern.match(file.name)
        if match:
            version_num = int(match.group(1))
            existing_versions.append((version_num, file))

    # Sort by version number
    existing_versions.sort(key=lambda x: x[0])

    # Determine paths
    if existing_versions:
        # Load from the latest version
        latest_version, latest_file = existing_versions[-1]
        knowledge_base_path = latest_file
        # Create next version
        next_version = latest_version + 1
    else:
        # No existing versions, start fresh
        knowledge_base_path = None
        next_version = 1

    # Create new version file with timestamp for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_knowledge_base_path = kb_folder / f"{prefix}_v{next_version}_{timestamp}.pkl"

    # Also create a "latest" symlink for convenience
    latest_symlink = kb_folder / f"{prefix}_latest.pkl"
    try:
        if knowledge_base_path is not None:
            # Only update symlink if there was a previous version
            if latest_symlink.exists() or latest_symlink.is_symlink():
                latest_symlink.unlink()
            latest_symlink.symlink_to(new_knowledge_base_path.name)
    except OSError:
        # Symlink creation might fail on some systems, that's okay
        pass

    # Clean up old versions (keep only the most recent N)
    cleanup_old_knowledge_base_versions(kb_folder, prefix, keep_last_n=DEFAULT_KEEP_LAST_N)

    return knowledge_base_path, new_knowledge_base_path


class CoSTEERSettings(ExtendedBaseSettings):
    """CoSTEER settings, this setting is supposed not to be used directly!!!"""

    class Config:
        env_prefix = "CoSTEER_"

    coder_use_cache: bool = False
    """Indicates whether to use cache for the coder"""

    max_loop: int = 10
    """Maximum number of task implementation loops"""

    fail_task_trial_limit: int = 20

    v1_query_former_trace_limit: int = 3
    v1_query_similar_success_limit: int = 3

    v2_query_component_limit: int = 1
    v2_query_error_limit: int = 1
    v2_query_former_trace_limit: int = 3
    v2_add_fail_attempt_to_latest_successful_execution: bool = False
    v2_error_summary: bool = False
    v2_knowledge_sampler: float = 1.0

    knowledge_base_path: Union[str, None] = None
    """Path to the knowledge base"""

    new_knowledge_base_path: Union[str, None] = None
    """Path to the new knowledge base"""

    enable_filelock: bool = False
    filelock_path: Union[str, None] = None

    max_seconds_multiplier: int = 10**6


CoSTEER_SETTINGS = CoSTEERSettings()
