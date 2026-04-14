import json
import re
from pathlib import Path

from rdagent.core.experiment import FBWorkspace
from rdagent.utils.env import Env


def get_runtime_environment_by_env(env: Env) -> str:
    implementation = FBWorkspace()
    fname = "runtime_info.py"
    implementation.inject_files(**{fname: (Path(__file__).absolute().resolve().parent / "runtime_info.py").read_text()})

    # Use the python binary from the environment's bin_path instead of just "python"
    # This ensures we use the correct python interpreter
    bin_path = env.conf.bin_path if hasattr(env.conf, "bin_path") else ""
    python_bin = f"{bin_path}/python" if bin_path else "python"

    stdout = implementation.execute(env=env, entry=f"{python_bin} {fname}")
    # Extract JSON from stdout (skip CUDA/container warnings)
    json_match = re.search(r"\{.*\}", stdout, re.DOTALL)
    if json_match is None:
        # Fallback: return empty JSON if parsing fails
        return json.dumps({}, indent=2)
    return json.dumps(json.loads(json_match.group()), indent=2)


def check_runtime_environment(env: Env) -> str:
    implementation = FBWorkspace()
    # 1) Check if strace exists in env
    strace_check = implementation.execute(env=env, entry="which strace || echo MISSING").strip()
    if strace_check.endswith("MISSING"):
        raise RuntimeError("`strace` not found in the target environment.")

    # 2) Check if coverage module works in env
    coverage_check = implementation.execute(env=env, entry="python -m coverage --version || echo MISSING").strip()
    if coverage_check.endswith("MISSING"):
        raise RuntimeError("`coverage` module not found or not runnable in the target environment.")
