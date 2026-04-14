from __future__ import annotations

import asyncio
from pathlib import Path

import fire

from rdagent.app.qlib_rd_loop.conf import ModelBasePropSetting
from rdagent.app.qlib_rd_loop.sota_conf import SOTA_PROP_SETTING
from rdagent.app.qlib_rd_loop.sota_proposal import SotaModelHypothesisGen
from rdagent.components.coder.CoSTEER.config import resolve_knowledge_base_paths
from rdagent.components.workflow.rd_loop import RDLoop
from rdagent.core.exception import ModelEmptyError
from rdagent.core.utils import import_class
from rdagent.log import rdagent_logger as logger


class SotaRDLoop(RDLoop):
    skip_loop_error = (ModelEmptyError,)

    def __init__(self, PROP_SETTING: ModelBasePropSetting) -> None:
        scen = import_class(PROP_SETTING.scen)()
        logger.log_object(scen, tag="scenario")
        logger.log_object(PROP_SETTING.model_dump(), tag="RDLOOP_SETTINGS")

        self.hypothesis_gen = None
        self.hypothesis2experiment = None
        self.coder = None
        self.runner = None
        self.summarizer = None
        self.trace = None

        from rdagent.core.conf import RD_AGENT_SETTINGS

        logger.log_object(RD_AGENT_SETTINGS.model_dump(), tag="RD_AGENT_SETTINGS")

        self._init_hypothesis_gen(scen, PROP_SETTING)
        self._init_components(scen, PROP_SETTING)

        from rdagent.core.proposal import Trace

        self.trace = Trace(scen=scen)

        from rdagent.utils.workflow import LoopBase, LoopMeta

        LoopBase.__init__(self)

    def _init_hypothesis_gen(self, scen, PROP_SETTING) -> None:
        kb_path = getattr(PROP_SETTING, "knowledge_base_path", None)
        if kb_path:
            kb_path = Path(kb_path)
        else:
            kb_path = self._resolve_kb_path(PROP_SETTING)
        sota_csv_path = getattr(PROP_SETTING, "sota_csv_path", None)
        self.hypothesis_gen = SotaModelHypothesisGen(scen, sota_csv_path=sota_csv_path, kb_path=kb_path)

    def _init_components(self, scen, PROP_SETTING) -> None:
        if hasattr(PROP_SETTING, "hypothesis2experiment") and PROP_SETTING.hypothesis2experiment:
            self.hypothesis2experiment = import_class(PROP_SETTING.hypothesis2experiment)()
        if hasattr(PROP_SETTING, "coder") and PROP_SETTING.coder:
            self.coder = import_class(PROP_SETTING.coder)(scen)
        if hasattr(PROP_SETTING, "runner") and PROP_SETTING.runner:
            self.runner = import_class(PROP_SETTING.runner)(scen)
        if hasattr(PROP_SETTING, "summarizer") and PROP_SETTING.summarizer:
            self.summarizer = import_class(PROP_SETTING.summarizer)(scen)

    @staticmethod
    def _resolve_kb_path(PROP_SETTING) -> Path | None:
        try:
            from rdagent.components.coder.model_coder.conf import MODEL_COSTEER_SETTINGS

            settings = MODEL_COSTEER_SETTINGS
            if getattr(settings, "auto_knowledge_versioning", False):
                kb_path, _ = resolve_knowledge_base_paths(settings)
                return kb_path
            elif settings.knowledge_base_path:
                return Path(settings.knowledge_base_path)
        except Exception as e:
            logger.warning(f"Could not resolve KB path: {e}")
        return None


def main(
    path=None,
    step_n: int | None = None,
    loop_n: int | None = None,
    all_duration: str | None = None,
    checkout: bool = True,
):
    if path is None:
        sota_loop = SotaRDLoop(SOTA_PROP_SETTING)
    else:
        sota_loop = SotaRDLoop.load(path, checkout=checkout)
    asyncio.run(sota_loop.run(step_n=step_n, loop_n=loop_n, all_duration=all_duration))


if __name__ == "__main__":
    fire.Fire(main)
