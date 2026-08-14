"""research-fleet: auditable, containerized, multi-GPU agent fleet.

Two entry points, one implementation:

    CLI     fleet run "..." --agents 4
    Python  from research_fleet import Fleet
"""

from .budget import (  # noqa: F401
    MODEL_COSTS,
    BudgetExceeded,
    BudgetTracker,
    ModelCost,
    Quote,
    Usage,
    cost_menu,
    quote,
)
from .config import FleetConfig, load_config  # noqa: F401
from .fleet import Fleet, RunReport, WorkflowReport  # noqa: F401
from .ledger import Ledger, Redactor  # noqa: F401
from .policy import ContainerPolicy, Policy  # noqa: F401
from .scheduler import Scheduler  # noqa: F401
from .spec import AgentConfig, JobKind, JobResult, JobSpec, JobState, Mount, Resources  # noqa: F401
from .workflow import Actor, Condition, Loop, Step, Workflow, WorkflowRunner  # noqa: F401

__version__ = "0.1.0"

__all__ = [
    "Fleet", "RunReport", "FleetConfig", "load_config",
    "JobSpec", "JobKind", "JobState", "JobResult", "AgentConfig", "Resources", "Mount",
    "Policy", "ContainerPolicy", "NetworkPolicy",
    "Ledger", "Redactor", "Scheduler",
    "BudgetTracker", "BudgetExceeded", "ModelCost", "Usage", "Quote",
    "MODEL_COSTS", "quote", "cost_menu",
    "Workflow", "Actor", "Step", "Loop", "Condition", "WorkflowRunner", "WorkflowReport",
    "__version__",
]
