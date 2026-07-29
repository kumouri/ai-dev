"""Coryphaeus — a small conductor LM that routes subtasks to a pool of workers.

An independent replication of the recipe in "Learning to Orchestrate Agents in Natural Language with
the Conductor" (Sakana AI, ICLR 2026, arXiv:2512.04388), extended with a world model over worker
success. See ``docs/RESEARCH.md``.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .datasets.loaders import Question, load_fixture, load_questions
from .orchestrate import ExecutionResult, execute, run_solo
from .reward import Score, extract_answer, score_answer, score_failure
from .rollout import RolloutRecord, group_advantages, one_rollout, rollout_group
from .schema import MAX_STEPS, SELF, Step, Workflow, WorkflowError
from .workflow import parse

__all__ = [
    "MAX_STEPS",
    "SELF",
    "ExecutionResult",
    "Question",
    "RolloutRecord",
    "Score",
    "Step",
    "Workflow",
    "WorkflowError",
    "__version__",
    "execute",
    "extract_answer",
    "group_advantages",
    "load_fixture",
    "load_questions",
    "one_rollout",
    "parse",
    "rollout_group",
    "run_solo",
    "score_answer",
    "score_failure",
]
