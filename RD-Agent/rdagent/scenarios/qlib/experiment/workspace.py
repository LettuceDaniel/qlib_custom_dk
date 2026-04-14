import re
from pathlib import Path
from typing import Any

import pandas as pd

from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.components.coder.model_coder.conf import MODEL_COSTEER_SETTINGS
from rdagent.core.experiment import FBWorkspace
from rdagent.log import rdagent_logger as logger
from rdagent.utils.env import QlibCondaConf, QlibCondaEnv, QTDockerEnv


class QlibFBWorkspace(FBWorkspace):
    def __init__(self, template_folder_path: Path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.inject_code_from_folder(template_folder_path)

    def execute(self, qlib_config_name: str = "conf.yaml", run_env: dict = {}, *args, **kwargs) -> str:
        if MODEL_COSTEER_SETTINGS.env_type == "docker":
            qtde = QTDockerEnv()
        elif MODEL_COSTEER_SETTINGS.env_type == "conda":
            qtde = QlibCondaEnv(conf=QlibCondaConf())
        else:
            logger.error(f"Unknown env_type: {MODEL_COSTEER_SETTINGS.env_type}")
            return None, "Unknown environment type"
        qtde.prepare()

        import os

        from rdagent.core.conf import RD_AGENT_SETTINGS

        mlruns_dir = RD_AGENT_SETTINGS.RDAGENT_MLRUNS_DIR
        sqlite_uri = f"sqlite:///{mlruns_dir}/mlflow.db"
        artifact_root = f"file://{mlruns_dir}/artifacts"

        os.environ.setdefault("RDAGENT_MLRUNS_DIR", mlruns_dir)
        os.environ.setdefault("RDAGENT_MODEL_DATA_DIR", RD_AGENT_SETTINGS.RDAGENT_MODEL_DATA_DIR)
        os.environ.setdefault("MLFLOW_TRACKING_URI", sqlite_uri)
        os.environ.setdefault("MLFLOW_ARTIFACT_ROOT", RD_AGENT_SETTINGS.MLFLOW_ARTIFACT_ROOT or artifact_root)
        run_env = {**run_env}

        # Unified H5 data folder
        h5_data_folder = Path("/workspace/qlib_data/h5_data")
        h5_data_folder.mkdir(parents=True, exist_ok=True)
        h5_all_path = h5_data_folder / "daily_pv_all.h5"

        # Run generate.py if daily_pv_all.h5 doesn't exist
        if not h5_all_path.exists():
            logger.info("daily_pv_all.h5 not found, running generate.py...")
            self._run_generate_py()

        # Run the Qlib backtest
        execute_qlib_log = qtde.check_output(
            local_path=str(self.workspace_path),
            entry=f"qrun {qlib_config_name}",
            env=run_env,
        )
        logger.log_object(execute_qlib_log, tag="Qlib_execute_log")

        execute_log = qtde.check_output(
            local_path=str(self.workspace_path),
            entry="python read_exp_res.py",
            env=run_env,
        )

        quantitative_backtesting_chart_path = self.workspace_path / "ret.pkl"
        if quantitative_backtesting_chart_path.exists():
            ret_df = pd.read_pickle(quantitative_backtesting_chart_path)
            logger.log_object(ret_df, tag="Quantitative Backtesting Chart")
        else:
            logger.error("No result file found.")
            return None, execute_qlib_log

        qlib_res_path = self.workspace_path / "qlib_res.csv"
        if qlib_res_path.exists():
            # Here, we ensure that the qlib experiment has run successfully before extracting information from execute_qlib_log using regex; otherwise, we keep the original experiment stdout.
            pattern = r"(Epoch\d+: train -[0-9\.]+, valid -[0-9\.]+|best score: -[0-9\.]+ @ \d+ epoch)"
            matches = re.findall(pattern, execute_qlib_log)
            execute_qlib_log = "\n".join(matches)
            return pd.read_csv(qlib_res_path, index_col=0).iloc[:, 0], execute_qlib_log
        else:
            logger.error(f"File {qlib_res_path} does not exist.")
            return None, execute_qlib_log

    def _run_generate_py(self):
        """Run generate.py to create daily_pv_all.h5"""
        from rdagent.utils.env import QlibCondaConf, QlibCondaEnv

        generate_script = Path(__file__).parent / "factor_data_template" / "generate.py"

        env = QlibCondaEnv(conf=QlibCondaConf(provider_uri="/workspace/qlib_data/us_data", region="us"))
        env.prepare()

        logger.info("Running generate.py to create H5 data...")
        output = env.check_output(
            local_path=str(generate_script.parent),
            entry="python generate.py",
        )
        logger.info(f"generate.py output:\n{output}")
