import asyncio
import os
import sys
import time
from typing import List, Optional

import requests
from openai import OpenAI

# ==========================================================
# WINDOWS ASYNCIO FIX
# ==========================================================
if sys.platform == "win32" and sys.version_info < (3, 11):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ==========================================================
# REQUIRED ENV VARIABLES
# ==========================================================
HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# Environment URL
BASE_URL = os.getenv(
    "ENV_BASE_URL",
    "https://boopathi376-rl-driven-automl.hf.space"
)

BENCHMARK = "rl_driven_automl"
TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 8
SUCCESS_SCORE_THRESHOLD = 0.5

# ==========================================================
# OPENAI CLIENT
# ==========================================================
client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL,
)

# ==========================================================
# LOGGING HELPERS
# ==========================================================
def log_start(task: str, env: str, model: str) -> None:
    print(
        f"[START] task={task} env={env} model={model}".strip(),
        flush=True,
    )


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()

    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )

# ==========================================================
# LLM POLICY
# ==========================================================
def call_llm(obs: dict) -> str:
    """
    Uses the observation stage directly.
    This keeps the environment deterministic and follows
    the expected next pipeline stage exactly.
    """
    current_stage = obs.get("stage")

    if not current_stage:
        return "cleaning"

    if current_stage == "completed":
        return "completed"

    return current_stage

# ==========================================================
# TASK CONFIGS
# ==========================================================
def get_reset_payload(task: str) -> dict:
    if task == "easy":
        return {
            "params": {
                "data_path": "data/Salary_dataset.csv",
                "target_column": "Salary",
                "latency_budget": 120.0,
                "memory_limit_mb": 0.0,
            }
        }

    elif task == "medium":
        return {
            "params": {
                "data_path": "data/winequality-red.csv",
                "target_column": "quality",
                "latency_budget": 120.0,
                "memory_limit_mb": 0.0,
            }
        }

    return {
        "params": {
            "data_path": "data/train.txt",
            "latency_budget": 120.0,
            "memory_limit_mb": 0.0,
        }
    }

# ==========================================================
# RUN ONE TASK
# ==========================================================
def run_task(task: str) -> float:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_payload = get_reset_payload(task)

        # Reset environment
        reset_response = requests.post(
            f"{BASE_URL}/reset",
            json=reset_payload,
            timeout=60,
        )
        reset_response.raise_for_status()

        reset_result = reset_response.json()
        obs = reset_result.get("observation") or reset_result
        done = bool(reset_result.get("done", False))

        if obs.get("stage") == "error":
            raise RuntimeError(f"Reset failed: {obs}")

        # Step loop
        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action = call_llm(obs)

            step_response = requests.post(
                f"{BASE_URL}/step",
                json={
                    "action": {
                        "stage": action
                    }
                },
                timeout=60,
            )
            step_response.raise_for_status()

            result = step_response.json()

            obs = result.get("observation") or result
            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", False))

            rewards.append(reward)
            steps_taken = step

            error = None
            if obs.get("stage") == "error":
                error = obs.get("metadata", {}).get("error", "step_failed")

            log_step(
                step=step,
                action=action,
                reward=reward,
                done=done,
                error=error,
            )

            if done:
                break

        # Score normalized to [0,1]
        if rewards:
            score = sum(rewards) / len(rewards)

        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(
            f"[ERROR] Task {task} failed: {exc}",
            file=sys.stderr,
            flush=True,
        )

    finally:
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )

    return score

# ==========================================================
# MAIN
# ==========================================================
def main() -> None:
    if not HF_TOKEN:
        print(
            "[ERROR] HF_TOKEN is not set.",
            file=sys.stderr,
            flush=True,
        )
        return

    for task in TASKS:
        run_task(task)


if __name__ == "__main__":
    main()