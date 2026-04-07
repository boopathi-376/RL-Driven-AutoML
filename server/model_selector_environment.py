import pandas as pd
import numpy as np
import time
from typing import Optional, Dict, Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# Data Models
try:
    # Try absolute import first (standard for Docker with PYTHONPATH=/app)
    from models import ModelSelectorAction, ModelSelectorObservation, EnvInput
except (ImportError, ValueError, ModuleNotFoundError):
    try:
        # Fallback to relative if part of a package structure
        from ..models import ModelSelectorAction, ModelSelectorObservation, EnvInput
    except (ImportError, ValueError, ModuleNotFoundError):
        # Final fallback for local development runs
        from models import ModelSelectorAction, ModelSelectorObservation, EnvInput

# Steps 8 Modules
from .steps_8.data_cleaning import DataCleaner, CleaningConfig
from .steps_8.encoding import Encoder, EncodingConfig
from .steps_8.feature_engineering import FeatureEngineer, FeatureEngineeringConfig
from .steps_8.scaling import Scaler, ScalingConfig
from .steps_8.feature_selection import FeatureSelector, FeatureSelectionConfig
from .steps_8.model_selection import SmartModelSelector, ModelSelectionConfig
from .steps_8.hyperparameter_tuning import HyperparameterTuner, TuningConfig
from .steps_8.ensemble import EnsembleBuilder, EnsembleConfig
from sklearn.feature_extraction.text import TfidfVectorizer

# ==========================================================
# GLOBAL SESSION STORE
# ==========================================================
_SESSION = {
    "X": None,
    "y": None,
    "config": None,
    "task_type": None,
    "pipeline_decisions": {},
    "components_reports": {},
    "current_stage_idx": 0,
    "selected_model": None,
    "prev_score": 0.0,
    "episode_id": None,
    "step_count": 0,
    "stages": None,
    "file_type":None
}

class ModelSelectorEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.stages = [
            "cleaning", "encoding", "engineering", "scaling", 
            "selection", "model_select", "tuning", "ensemble"
        ]
        self._sync_with_global()

    def _sync_with_global(self):
        self.X = _SESSION["X"]
        self.y = _SESSION["y"]
        self.config = _SESSION["config"]
        self.task_type = _SESSION["task_type"]
        self.pipeline_decisions = _SESSION["pipeline_decisions"]
        self.components_reports = _SESSION["components_reports"]
        self.current_stage_idx = _SESSION["current_stage_idx"]
        self.selected_model = _SESSION["selected_model"]
        self.prev_score = _SESSION["prev_score"]
        self.stages = _SESSION["stages"] or [
            "cleaning", "encoding", "engineering", "scaling",
            "selection", "model_select", "tuning", "ensemble"
        ]
        self.file_type = _SESSION["file_type"]
        # Restore state object
        if _SESSION["episode_id"]:
            self._state = State(
                episode_id=_SESSION["episode_id"],
                step_count=_SESSION["step_count"]
            )

    def _save_to_global(self):
        _SESSION.update({
            "X": self.X, "y": self.y, "config": self.config,
            "task_type": self.task_type, "pipeline_decisions": self.pipeline_decisions,
            "components_reports": self.components_reports, "current_stage_idx": self.current_stage_idx,
            "selected_model": self.selected_model, "prev_score": self.prev_score,
            "episode_id": self._state.episode_id,
            "step_count": self._state.step_count,
            "stages": self.stages,
            "file_type": self.file_type
        })

    def _reset_internal_vars(self):
        _SESSION.update({
            "X": None, "y": None, "config": None, "task_type": None,
            "pipeline_decisions": {}, "components_reports": {},
            "current_stage_idx": 0, "selected_model": None, "prev_score": 0.0,
            "episode_id": None, "step_count": 0,
            "stages": None,
            "file_type": None
        })
        self._sync_with_global()

    def reset(self, params: Optional[EnvInput] = None) -> ModelSelectorObservation:
        self._reset_internal_vars()
        self._state = State(episode_id=str(uuid4()), step_count=0)

        print(f"DEBUG: Reset params received: {params}")

        

        try:
            # Store config directly from EnvInput model
            self.config = params

            if not self.config.data_path:
                return self._error("data_path is required")

            file_path = self.config.data_path.lower()
            clean_path = self.config.data_path.replace("\\", "/")

            # --------------------------------------------------
            # Detect file type and choose stages
            # --------------------------------------------------
            if file_path.endswith(".txt"):
                self.file_type = "text"
                self.stages = [
                    "cleaning",
                    "model_select",
                    "tuning",
                    "ensemble"
                ]

            elif file_path.endswith(".csv"):
                preview = pd.read_csv(clean_path, nrows=5)

                # Single-column CSV = treat as text dataset
                if len(preview.columns) == 1:
                    self.file_type = "text"
                    self.stages = [
                        "cleaning",
                        "model_select",
                        "tuning",
                        "ensemble"
                    ]
                else:
                    self.file_type = "structured"
                    self.stages = [
                        "cleaning",
                        "encoding",
                        "engineering",
                        "scaling",
                        "selection",
                        "model_select",
                        "tuning",
                        "ensemble"
                    ]

            elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
                self.file_type = "structured"
                self.stages = [
                    "cleaning",
                    "encoding",
                    "engineering",
                    "scaling",
                    "selection",
                    "model_select",
                    "tuning",
                    "ensemble"
                ]

            else:
                return self._error(f"Unsupported file type: {file_path}")

            # --------------------------------------------------
            # Load dataset
            # --------------------------------------------------
            if self.file_type == "text":
                if clean_path.endswith(".txt"):
                    with open(clean_path, "r", encoding="utf-8") as f:
                        text_data = f.readlines()

                    data = pd.DataFrame({"text": text_data})

                else:
                    data = pd.read_csv(clean_path)

                    # Rename first column to text for consistency
                    first_col = data.columns[0]
                    data = data.rename(columns={first_col: "text"})

                self.X = data

                # Use dummy labels for text datasets
                self.y = np.zeros(len(self.X))

            else:
                if clean_path.endswith(".csv"):
                    data = pd.read_csv(clean_path)
                else:
                    data = pd.read_excel(clean_path)

                target_col = self.config.target_column or data.columns[-1]

                if target_col not in data.columns:
                    return self._error(f"Target column '{target_col}' not found")

                self.y = data[target_col]
                self.X = data.drop(columns=[target_col])

            self.task_type = self._infer_task()

            self._save_to_global()

            print(f"DEBUG file_type: {self.file_type}")
            print(f"DEBUG stages: {self.stages}")
            print(f"--- RESET SUCCESS: Loaded {self.X.shape} ---")

            return self._build_obs(done=False, reward=0.0)

        except Exception as e:
            return self._error(f"Reset failed: {str(e)}")
    def step(self, action: ModelSelectorAction) -> ModelSelectorObservation:
        self._sync_with_global()

        if self.X is None:
            return self._error("Session lost. Reset required.")

        current_stage = self.stages[self.current_stage_idx]

        if action.stage != current_stage:
            return self._error(f"Sync error: Expected {current_stage}")

        start_time = time.time()

        try:
            # --- 1. Module Selection ---
            if current_stage == "cleaning":
                obj = DataCleaner(CleaningConfig())

                self.X = obj.clean(self.X)
                report = obj.get_report()

                # Sync y to rows that survived cleaning.
                # y can be a pandas Series (structured data) or numpy array (text data).
                if self.y is not None:
                    surviving_idx = self.X.index
                    if isinstance(self.y, pd.Series):
                        self.y = self.y.loc[surviving_idx].reset_index(drop=True)
                    else:
                        # numpy array — index directly with the surviving integer positions
                        self.y = self.y[surviving_idx]
                self.X = self.X.reset_index(drop=True)

                self.pipeline_decisions[current_stage] = report
                self.components_reports[current_stage] = report

            elif current_stage == "encoding":
                if self.X is None or not isinstance(self.X, pd.DataFrame):
                    return self._error("Invalid data before encoding stage")

                try:
                    n_rows, n_cols = self.X.shape
                except Exception as e:
                    return self._error(f"Shape extraction failed: {str(e)}")

                # Estimate memory usage
                estimated_size_gb = (n_rows * n_cols * 8) / (1024**3)

                if estimated_size_gb > 1.0 or n_cols > 1000:
                    print("Large dataset detected → switching to safe encoding")

                    config = EncodingConfig(
                        method="label",   # safer than one-hot
                    )
                else:
                    config = EncodingConfig()

                obj = Encoder(config)
                self.X = obj.fit_transform(self.X, self.y)

                # Reduce memory AFTER encoding
                try:
                    self.X = self.X.astype(np.float32)
                except:
                    pass

                # Hard feature cap (prevents explosion)
                if self.X.shape[1] > 1000:
                    self.X = self.X.iloc[:, :1000]

                report = obj.get_report()

            elif current_stage == "engineering":
                obj = FeatureEngineer(FeatureEngineeringConfig())
                self.X = obj.fit_transform(self.X)
                report = obj.get_report()

            elif current_stage == "scaling":
                obj = Scaler(ScalingConfig())
                self.X = obj.fit_transform(self.X)
                report = obj.get_report()

            elif current_stage == "selection":
                obj = FeatureSelector(FeatureSelectionConfig())
                self.X = obj.fit_transform(self.X, self.y)
                report = obj.get_report()

            elif current_stage == "model_select":
                if self.file_type == "text":
                    vectorizer = TfidfVectorizer(max_features=1000)
                    X_vec = vectorizer.fit_transform(self.X["text"])

                    self.selected_model = vectorizer
                    report = {
                        "selected_model": "TF-IDF Vectorizer",
                        "score": 0.5,
                        "train_score": 0.5,
                        "details": {
                            "n_features": X_vec.shape[1]
                        }
                    }
                else:
                    obj = SmartModelSelector(ModelSelectionConfig())
                    obj.fit(self.X, self.y)

                    self.selected_model = obj.get_model()
                    report = obj.get_report()

                    self.pipeline_decisions[current_stage] = {
                        "selected_model": report.get("selected_model"),
                        "decision_meta": report.get("decision_meta"),
                        "score": report.get("score")
                    }

            elif current_stage == "tuning":
                obj = HyperparameterTuner(TuningConfig())
                self.selected_model = obj.tune(self.selected_model, self.X, self.y)
                report = obj.get_report()

            elif current_stage == "ensemble":
                if self.file_type == "text":
                    report = {
                        "status": "skipped for text",
                        "applied": False,
                        "reason": "No supervised model available",
                        "score": 0.5,
                        "train_score": 0.5           
                    }
                else:
                    obj = EnsembleBuilder(EnsembleConfig())
                    self.selected_model = obj.build([self.selected_model], self.X, self.y)
                    report = obj.get_report()
                

            else:
                return self._error(f"Unknown stage: {current_stage}")

            # --- 2. Reward Calculation ---
            training_time = time.time() - start_time

            reward = self._compute_reward(
                train_score=report.get("train_score", 0.5),
                val_score=report.get("score", 0.5),
                model_name=str(type(self.selected_model).__name__) if self.selected_model is not None else "unknown",
                training_time=training_time,
                n_features=self.X.shape[1]
            )

            # --- 3. State Update ---
            self.pipeline_decisions[current_stage] = report
            self.components_reports[current_stage] = report
            self.current_stage_idx += 1
            self._state.step_count += 1

            self._save_to_global()

            done = self.current_stage_idx >= len(self.stages)

            if done:
                return self._terminal(reward, "Pipeline Complete")

            return self._build_obs(done=False, reward=reward)

        except Exception as e:
            return self._error(f"Failed at {current_stage}: {str(e)}")

    def _compute_reward(self, train_score, val_score, model_name, training_time, n_features) -> float:

        improvement = val_score - self.prev_score
        overfit_gap = max(0, train_score - val_score)
        
        reward = (
            val_score + (0.2 * improvement) - (0.3 * overfit_gap) 
            - (np.tanh(training_time / 10) * 0.2)
            + (0.1 * (1 / (1 + np.log1p(n_features))))
        )
        if self.task_type == "text_processing":
            return 0.3  # constant stable reward
        if self.task_type == "regression" and self.config.metric != "r2":
            val_score = -val_score  # convert RMSE to "higher is better"
        if "RandomForest" in model_name or "Boost" in model_name: reward -= 0.08
        if n_features > 500: reward -= 0.08
        if abs(train_score - val_score) < 0.05: reward += 0.05
        
        self.prev_score = val_score
        return float(np.clip(reward, -1, 1))

    def _build_obs(self, done: bool, reward: float) -> ModelSelectorObservation:
        stage_name = self.stages[self.current_stage_idx] if not done else "completed"
        profile = {"n_samples": self.X.shape[0], "n_features": self.X.shape[1]}
        
        return ModelSelectorObservation(
            stage=stage_name,
            task_type=self.task_type or "unknown",
            dataset_profile=profile,
            partial_pipeline=self.pipeline_decisions,
            latency_budget=float(self.config.latency_budget or 0.0),
            memory_limit_mb=float(self.config.memory_limit_mb or 0.0),
            reward=reward,
            done=done,
            progress=float(self.current_stage_idx / len(self.stages)),
            metadata={"reports": self.components_reports}
        )

    def _infer_task(self) -> str:
        if self.y is None:
            return "text_processing"
        # self.y may be a numpy array (text datasets use np.zeros dummy labels)
        # or a pandas Series (structured datasets). Wrap in Series to safely call .nunique().
        y_series = pd.Series(self.y)
        if y_series.nunique() == 0:
            return "text_processing"
        return "classification" if y_series.dtype == 'object' or y_series.nunique() < 20 else "regression"
    
    def _terminal(self, reward, reason):
        obs = self._build_obs(done=True, reward=reward)
        obs.metadata["reason"] = reason
        return obs

    def _error(self, msg: str) -> ModelSelectorObservation:
        print(f"--- ENVIRONMENT ERROR: {msg} ---")
        return ModelSelectorObservation(
            stage="error",
            task_type="unknown",           # Mandatory field
            dataset_profile={},            # Mandatory field
            partial_pipeline={},           # Mandatory field
            latency_budget=0.0,            # Mandatory field
            memory_limit_mb=0.0,           # Mandatory field
            progress=0.0,                  # Mandatory field
            reward=-1.0,                   # Provided
            done=True,                     # Provided
            metadata={"error": str(msg)}   # Provided
        )

    @property
    def state(self) -> dict:
        """Returns a detailed state dictionary for the /state endpoint."""
        # Calculate current and next stage names
        idx = self.current_stage_idx
        stages_count = len(self.stages)
        
        current_stage = self.stages[idx] if idx < stages_count else "completed"
        
        next_idx = idx + 1
        next_stage = self.stages[next_idx] if next_idx < stages_count else None

        return {
            "episode_id": self._state.episode_id,
            "step_count": self._state.step_count,
            "current_step": idx + 1,
            "total_steps": stages_count,
            "current_stage": current_stage,
            "next_stage": next_stage,
            "task_type": self.task_type
        }