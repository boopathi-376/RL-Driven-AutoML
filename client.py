
from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import ModelSelectorAction, ModelSelectorObservation, EnvInput
except (ImportError, ValueError):
    from models import ModelSelectorAction, ModelSelectorObservation, EnvInput


class ModelSelectorEnv(
    EnvClient[ModelSelectorAction, ModelSelectorObservation, State]
):
    def _step_payload(self, action: ModelSelectorAction) -> Dict:
        return {
            "stage": action.stage,
        }

    def _reset_payload(self, **kwargs) -> Dict:
        return {
            "params": kwargs
        }

    def _parse_result(self, payload: Dict) -> StepResult[ModelSelectorObservation]:
        obs_data = payload.get("observation", {})

        observation = ModelSelectorObservation(
            stage=obs_data.get("stage", "unknown"),
            task_type=obs_data.get("task_type", "unknown"),
            dataset_profile=obs_data.get("dataset_profile", {}),
            available_choices=obs_data.get("available_choices", []),
            partial_pipeline=obs_data.get("partial_pipeline", {}),
            latency_budget=float(obs_data.get("latency_budget", 0.0)),
            memory_limit_mb=float(obs_data.get("memory_limit_mb", 0.0)),
            progress=float(obs_data.get("progress", 0.0)),
            reward=float(payload.get("reward", 0.0)),
            done=bool(payload.get("done", False)),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=float(payload.get("reward", 0.0)),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
