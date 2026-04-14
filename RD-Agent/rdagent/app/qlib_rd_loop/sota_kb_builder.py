from __future__ import annotations

from rdagent.components.coder.CoSTEER.knowledge_management import CoSTEERKnowledgeBaseV2


def sync_sota_tags(kb: CoSTEERKnowledgeBaseV2, sota_run_ids: list[str]) -> int:
    run_id_to_task_info: dict[str, str] = {v: k for k, v in kb.success_task_run_id.items()}
    matched = 0
    for run_id in sota_run_ids:
        task_info = run_id_to_task_info.get(run_id)
        if task_info is not None:
            kb.sota_task_info_set.add(task_info)
            matched += 1
    return matched
