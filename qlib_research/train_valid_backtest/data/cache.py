import os
import importlib.util
import inspect
from pathlib import Path
import pandas as pd

_h5_cache: dict[str, pd.DataFrame] = {}


def get_h5_data(h5_path: str) -> pd.DataFrame:
    abs_path = str(Path(h5_path).resolve())
    if abs_path not in _h5_cache:
        print(f"  [H5 Cache] Loading {abs_path} ...")
        df = pd.read_hdf(abs_path)
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        _h5_cache[abs_path] = df
    # Return reference for memory efficiency.
    # NOTE: Callers MUST NOT mutate the returned DataFrame in-place.
    return _h5_cache[abs_path]


_model_module_cache: dict[str, object] = {}
_model_class_cache: dict[str, type] = {}


def _load_model_module(model_folder: str):
    abs_path = str(Path(model_folder).resolve())
    if abs_path not in _model_module_cache:
        model_path = os.path.join(abs_path, "model_architecture.py")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model architecture file not found: {model_path}")
        module_name = f"model_mod_{Path(abs_path).name}"
        spec = importlib.util.spec_from_file_location(module_name, model_path)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        _model_module_cache[abs_path] = model_module
        _model_class_cache[abs_path] = model_module.model_cls
    return _model_module_cache[abs_path]


def load_model_class(model_folder: str) -> type:
    _load_model_module(model_folder)
    abs_path = str(Path(model_folder).resolve())
    return _model_class_cache[abs_path]


def load_model_module(model_folder: str):
    _load_model_module(model_folder)
    abs_path = str(Path(model_folder).resolve())
    return _model_module_cache[abs_path]


def create_model_instance(model_info, model_class, model_config):
    """Model instantiation from checkpoint + config.

    Args:
        model_info: state_dict or checkpoint dict
        model_class: model class type
        model_config: model_config.json dict

    Returns:
        model: instantiated and loaded model (on CPU, eval mode)
    """
    import torch

    sig = inspect.signature(model_class.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    kwargs = {k: v for k, v in model_config.items() if k in valid_params}

    model = model_class(**kwargs)
    state = model_info.get("model_state_dict", model_info) if isinstance(model_info, dict) else model_info
    model.load_state_dict(state)
    model.eval()
    return model


def get_model_params(model_folder: str, model_config: dict) -> int:
    """Get total model parameters using existing load_model_class cache.

    Args:
        model_folder: model directory path
        model_config: model config dict (for kwargs)

    Returns:
        int: total parameters, or 0 if not available
    """
    try:
        model_module = load_model_module(model_folder)
        if not hasattr(model_module, "get_total_params"):
            return 0
        sig = inspect.signature(model_module.model_cls.__init__)
        valid_params = set(sig.parameters.keys()) - {"self"}
        kwargs = {k: v for k, v in model_config.items() if k in valid_params}
        return model_module.get_total_params(**kwargs)
    except Exception:
        return 0
