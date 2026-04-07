from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from openenv.core.env_server.types import Action, Observation

# =========================
# CONFIG / INPUT
# =========================
class EnvInput(BaseModel):
    goal: Optional[str] = None
    data_path: Optional[str] = None
    data_type: Optional[str] = None
    target_column: Optional[str] = None
    latency_budget: Optional[float] = None
    memory_limit_mb: Optional[float] = None
    explainability_level: Optional[str] = None
    metric: Optional[str] = None

# =========================
# ACTION
# =========================
class ModelSelectorAction(BaseModel):
    # This must match the current stage the Environment is in
    stage: str  

# =========================
# OBSERVATION
# =========================



class ModelSelectorObservation(Observation):
    stage: str
    task_type: str
    dataset_profile: Dict[str, Any]
    partial_pipeline: Dict[str, Any]

    latency_budget: float
    memory_limit_mb: float
    progress: float

    available_choices: List[str] = Field(default_factory=list)

    reward: Optional[float] = None
    done: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)