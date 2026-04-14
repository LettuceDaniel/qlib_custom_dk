from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

from rdagent.components.coder.CoSTEER.knowledge_management import CoSTEERKnowledgeBaseV2
from rdagent.core.proposal import Hypothesis, Scenario, Trace
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.qlib.proposal.model_proposal import QlibModelHypothesisGen


class SotaModelHypothesisGen(QlibModelHypothesisGen):
    def __init__(self, scen: Scenario, sota_csv_path: str | None = None, kb_path: str | Path | None = None) -> None:
        super().__init__(scen)
        self.sota_knowledge_list: list = []
        self._loaded = False
        self._sota_csv_path = sota_csv_path
        self._kb_path = Path(kb_path) if kb_path else None

    def _load_sota_knowledge(self) -> None:
        if self._loaded:
            return
        self._loaded = True

        if self._kb_path is None or not self._kb_path.exists():
            logger.warning(f"KB path not found: {self._kb_path}")
            return

        kb: CoSTEERKnowledgeBaseV2 = pickle.load(open(self._kb_path, "rb"))
        if not isinstance(kb, CoSTEERKnowledgeBaseV2):
            logger.warning("Loaded KB is not CoSTEERKnowledgeBaseV2")
            return

        sota_set = getattr(kb, "sota_task_info_set", set())

        if not sota_set and self._sota_csv_path:
            from rdagent.app.qlib_rd_loop.extract_sota import extract_sota_run_ids
            from rdagent.app.qlib_rd_loop.sota_kb_builder import sync_sota_tags

            sota_run_ids = extract_sota_run_ids(self._sota_csv_path)
            if sota_run_ids:
                matched = sync_sota_tags(kb, sota_run_ids)
                logger.info(f"SOTA sync: {matched}/{len(sota_run_ids)} run_ids matched")
                sota_set = kb.sota_task_info_set

        for task_info in sota_set:
            knowledge = kb.success_task_to_knowledge_dict.get(task_info)
            if knowledge is not None:
                self.sota_knowledge_list.append(knowledge)

        logger.info(f"Loaded {len(self.sota_knowledge_list)} SOTA knowledge entries")

    def _format_sota_rag(self) -> str:
        if not self.sota_knowledge_list:
            return ""

        parts = ["\n\n## Reference: SOTA Model Knowledge from Previous Runs\n"]
        for i, k in enumerate(self.sota_knowledge_list):
            task = k.target_task
            code = k.implementation.all_codes[:2000] if k.implementation else "N/A"
            parts.append(
                f"### SOTA Model {i + 1}: {getattr(task, 'name', 'Unknown')}\n"
                f"Description: {getattr(task, 'description', 'N/A')}\n"
                f"Architecture: {getattr(task, 'architecture', 'N/A')}\n"
                f"```\n{code}\n```\n"
            )
        return "\n".join(parts)

    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]:
        self._load_sota_knowledge()

        context_dict, json_flag = super().prepare_context(trace)

        sota_rag = self._format_sota_rag()
        if sota_rag:
            context_dict["RAG"] = context_dict.get("RAG", "") + sota_rag

        return context_dict, json_flag

    def convert_response(self, response: str) -> Hypothesis:
        return super().convert_response(response)
