import random
import re
import shutil
from pathlib import Path

import pandas as pd
from jinja2 import Environment, StrictUndefined

from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.log import rdagent_logger as logger
from rdagent.components.coder.model_coder.conf import MODEL_COSTEER_SETTINGS
from rdagent.utils.env import QTDockerEnv, QlibCondaEnv, QlibCondaConf


H5_DATA_PATH = Path("/workspace/qlib_data/h5_data")
H5_ALL_PATH = H5_DATA_PATH / "daily_pv_all.h5"
H5_DEBUG_PATH = H5_DATA_PATH / "daily_pv_debug.h5"


def generate_h5_data_from_qlib():
    template_path = Path(__file__).parent / "factor_data_template"

    if H5_ALL_PATH.exists() and H5_DEBUG_PATH.exists():
        logger.info("H5 data files already exist at /workspace/qlib_data/h5_data/")
    else:
        if MODEL_COSTEER_SETTINGS.env_type == "docker":
            qtde = QTDockerEnv()
        elif MODEL_COSTEER_SETTINGS.env_type == "conda":
            qtde = QlibCondaEnv(conf=QlibCondaConf())
        else:
            raise ValueError(f"Unknown env_type: {MODEL_COSTEER_SETTINGS.env_type}")
        qtde.prepare()

        execute_log = qtde.check_output(
            local_path=str(template_path),
            entry=f"python generate.py",
        )

        assert H5_ALL_PATH.exists(), (
            "daily_pv_all.h5 is not generated. It means rdagent/scenarios/qlib/experiment/factor_data_template/generate.py is not executed correctly. Please check the log: \n"
            + execute_log
        )
        assert H5_DEBUG_PATH.exists(), (
            "daily_pv_debug.h5 is not generated. It means rdagent/scenarios/qlib/experiment/factor_data_template/generate.py is not executed correctly. Please check the log: \n"
            + execute_log
        )


def get_file_desc(p: Path, variable_list=[]) -> str:
    """
    Get the description of a file based on its type.

    Parameters
    ----------
    p : Path
        The path of the file.

    Returns
    -------
    str
        The description of the file.
    """
    p = Path(p)

    JJ_TPL = Environment(undefined=StrictUndefined).from_string("""
# {{file_name}}

## File Type
{{type_desc}}

## Content Overview
{{content}}
""")

    if p.name.endswith(".h5"):
        df = pd.read_hdf(p)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_colwidth", None)

        df_info = "### Data Structure\n"
        df_info += (
            f"- Index: MultiIndex with levels {df.index.names}\n"
            if isinstance(df.index, pd.MultiIndex)
            else f"- Index: {df.index.name}\n"
        )

        df_info += "\n### Columns\n"
        columns = df.dtypes.to_dict()
        grouped_columns = {}

        for col in columns:
            if col.startswith("$"):
                prefix = col.split("_")[0] if "_" in col else col
                grouped_columns.setdefault(prefix, []).append(col)
            else:
                grouped_columns.setdefault("other", []).append(col)

        if variable_list:
            df_info += "#### Relevant Columns:\n"
            relevant_line = ", ".join(f"{col}: {columns[col]}" for col in variable_list if col in columns)
            df_info += relevant_line + "\n"
        else:
            df_info += "#### All Columns:\n"
            grouped_items = list(grouped_columns.items())
            random.shuffle(grouped_items)
            for prefix, cols in grouped_items:
                header = "Other Columns" if prefix == "other" else f"{prefix} Related Columns"
                df_info += f"\n#### {header}:\n"
                random.shuffle(cols)
                line = ", ".join(f"{col}: {columns[col]}" for col in cols)
                df_info += line + "\n"

        if "REPORT_PERIOD" in df.columns:
            one_instrument = df.index.get_level_values("instrument")[0]
            df_on_one_instrument = df.loc[pd.IndexSlice[:, one_instrument], ["REPORT_PERIOD"]]
            df_info += "\n### Sample Data\n"
            df_info += f"Showing data for instrument {one_instrument}:\n"
            df_info += str(df_on_one_instrument.head(5))

        return JJ_TPL.render(
            file_name=p.name,
            type_desc="HDF5 Data File",
            content=df_info,
        )

    elif p.name.endswith(".md"):
        with open(p) as f:
            content = f.read()
            return JJ_TPL.render(
                file_name=p.name,
                type_desc="Markdown Documentation",
                content=content,
            )

    else:
        raise NotImplementedError(
            f"file type {p.name} is not supported. Please implement its description function.",
        )


def get_data_statistics() -> dict:
    """
    Dynamically collect dataset statistics from H5 file and model config.

    Returns
    -------
    dict
        Dictionary containing:
        - data_frequency: str (e.g., "Daily")
        - total_data_points: int
        - train_data_points: int (estimated)
        - valid_data_points: int (estimated)
        - universe_size: int (number of instruments)
        - start_date: str
        - end_date: str
        - num_features: int
        - label_definition: str
        - label_normalization: str
        - prediction_horizon: str
    """
    # 1. Get data statistics from H5 file
    if not H5_ALL_PATH.exists():
        generate_h5_data_from_qlib()

    with pd.HDFStore(H5_ALL_PATH) as store:
        df = store["data"]

        dates = df.index.get_level_values("datetime")
        instruments = df.index.get_level_values("instrument")

        total_data_points = len(df)
        universe_size = instruments.nunique()
        start_date = dates.min().strftime("%Y-%m-%d")
        end_date = dates.max().strftime("%Y-%m-%d")

        # Estimate data frequency
        # Filter out 0-day diffs (same day data) and check minimum interval
        unique_diffs = dates.to_series().diff().dropna()
        unique_diffs = unique_diffs[unique_diffs > pd.Timedelta(0)]
        min_diff_days = unique_diffs.min().days
        if min_diff_days == 1:
            data_frequency = "Daily (with gaps for weekends/holidays)"
        elif min_diff_days == 7:
            data_frequency = "Weekly"
        else:
            data_frequency = f"Other (min interval: {min_diff_days} days)"

    # 2. Get model config information
    model_config_path = Path(__file__).parent / "model_template" / "conf_baseline_factors_model.yaml"
    num_features = None
    label_definition = "Ref($close, -5)/$close - 1 (5-day forward return)"
    label_normalization = "CSRankNorm"
    prediction_horizon = "5 days"

    if model_config_path.exists():
        with open(model_config_path) as f:
            content = f.read()
            # Extract num_features from pt_model_kwargs
            nf_match = re.search(r'"num_features":\s*(\d+)', content)
            if nf_match:
                num_features = int(nf_match.group(1))
            else:
                # Fallback: count feature names in config
                name_list_match = re.search(r"\[([\w\s,]+)\]\s*$", content, re.MULTILINE)
                if name_list_match:
                    names = [n.strip() for n in name_list_match.group(1).split(",")]
                    num_features = len([n for n in names if n])

    if num_features is None:
        num_features = 5  # fallback default

    # 3. Estimate train/valid split (based on default config)
    # Default: train_end = 2022-12-31, valid_end = 2024-12-31
    train_end = pd.Timestamp("2022-12-31")
    valid_end = pd.Timestamp("2024-12-31")
    test_end = pd.Timestamp(end_date)

    # Calculate approximate split
    train_data = df[df.index.get_level_values("datetime") <= train_end]
    valid_data = df[
        (df.index.get_level_values("datetime") > train_end) & (df.index.get_level_values("datetime") <= valid_end)
    ]

    train_data_points = len(train_data) if len(train_data) > 0 else int(total_data_points * 0.6)
    valid_data_points = len(valid_data) if len(valid_data) > 0 else int(total_data_points * 0.2)

    return {
        "data_frequency": data_frequency,
        "total_data_points": total_data_points,
        "train_data_points": train_data_points,
        "valid_data_points": valid_data_points,
        "universe_size": universe_size,
        "start_date": start_date,
        "end_date": end_date,
        "num_features": num_features,
        "label_definition": label_definition,
        "label_normalization": label_normalization,
        "prediction_horizon": prediction_horizon,
    }


def get_data_statistics_for_rag() -> str:
    """Get data statistics formatted for RAG."""
    stats = get_data_statistics()
    return f"""Dataset Information:
- Data Frequency: {stats["data_frequency"]}
- Total Data Points: {stats["total_data_points"]:,}
- Training Data Points: {stats["train_data_points"]:,}
- Validation Data Points: {stats["valid_data_points"]:,}
- Universe Size: {stats["universe_size"]} instruments
- Date Range: {stats["start_date"]} ~ {stats["end_date"]}
- Number of Factor Features: {stats["num_features"]}
- Label Definition: {stats["label_definition"]}
- Label Normalization: {stats["label_normalization"]}
- Prediction Horizon: {stats["prediction_horizon"]}

Data Characteristics:
- Type: Time-series (Daily frequency, contains weekend/holiday gaps)
- Label: Next-day return prediction (Alpha)"""


def get_hyperparameter_guidelines() -> str:
    """Get hyperparameter guidelines for model design and training."""
    return """Model Design Constraints (Baseline):
- Hidden Dim: >= 128 (>1M samples), >= 32 (<1M samples)
- Layers: Decide based on model type
- Lookback Window: 20 timesteps (step_len = num_timesteps)
- Regularization: Dropout(0.2-0.5) + Weight Decay(1e-5)
- Activation: ReLU/Leaky ReLU (Hidden), None/Linear (Final for CSRankNorm)
- Optimizer: AdamW (LR: 1e-3 or 5e-4)
- Batch Size: 512 or 1024
- Gradient Clipping: 1.0
- Early Stopping: Patience 5-8 epochs"""


def get_data_folder_intro(fname_reg: str = ".*", flags=0, variable_mapping=None) -> str:
    """
    Directly get the info of the data folder.
    It is for preparing prompting message.

    Parameters
    ----------
    fname_reg : str
        a regular expression to filter the file name.

    flags: str
        flags for re.match

    Returns
    -------
        str
            The description of the data folder.
    """

    if not H5_ALL_PATH.exists():
        # FIXME: (xiao) I think this is writing in a hard-coded way.
        # get data folder intro does not imply that we are generating the data folder.
        generate_h5_data_from_qlib()
    content_l = []
    for p in Path(FACTOR_COSTEER_SETTINGS.data_folder_debug).iterdir():
        if re.match(fname_reg, p.name, flags) is not None:
            if variable_mapping:
                content_l.append(get_file_desc(p, variable_mapping.get(p.stem, [])))
            else:
                content_l.append(get_file_desc(p))
    return "\n----------------- file splitter -------------\n".join(content_l)
