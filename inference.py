import asyncio
import os
import sys
from typing import List, Optional, Dict, Any

# ==========================================================
# WINDOWS ASYNCIO FIX (Prevents SSL / Loop closed errors)
# ==========================================================
if sys.platform == 'win32' and sys.version_info < (3, 11):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from openai import OpenAI

from client import ModelSelectorEnv
from models import ModelSelectorAction

API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

def check_keys():
    if not API_KEY:
        print("\n[ERROR] API key missing!")
        print("Please set your Hugging Face Token (with 'inference' access) or OpenAI key:")
        print("    $env:HF_TOKEN = \"your_token_here\"\n")
        return False
    return True
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

ENV_URL = os.getenv("ENV_URL") or "https://boopathi376-rl-driven-automl.hf.space"

TASK_NAME = "model_selector"
BENCHMARK = "openenv"
MAX_STEPS = 8
SUCCESS_SCORE_THRESHOLD = 0.5


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_text = error if error else "null"
    done_text = str(done).lower()

    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_text} error={error_text}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_text = ",".join([f"{r:.2f}" for r in rewards])

    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_text}",
        flush=True,
    )


def print_step_summary(step_outputs: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print("FINAL STEP OUTPUT SUMMARY")
    print("=" * 80)

    for item in step_outputs:
        print(f"\nStep {item['step']}")
        print(f"Stage        : {item['stage']}")
        print(f"Reward       : {item['reward']:.4f}")
        print(f"Done         : {item['done']}")
        print(f"Next Stage   : {item['next_stage']}")
        print(f"Progress     : {item['progress']:.2f}")

        if item["dataset_profile"]:
            print("Dataset Profile:")
            for key, value in item["dataset_profile"].items():
                print(f"  - {key}: {value}")

        if item["report"]:
            print("Returned Report:")
            for key, value in item["report"].items():
                print(f"  - {key}: {value}")
        else:
            print("Returned Report: None")

        if item["partial_pipeline"]:
            print("Pipeline Decisions:")
            for key, value in item["partial_pipeline"].items():
                print(f"  - {key}: {value}")


async def main() -> None:
    if not check_keys():
        return

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    env = ModelSelectorEnv(base_url=ENV_URL)

    rewards: List[float] = []
    step_outputs: List[Dict[str, Any]] = []

    steps_taken = 0
    score = 0.0
    success = False

    log_start(
        task=TASK_NAME,
        env=BENCHMARK,
        model=MODEL_NAME,
    )

    try:
        result = await env.reset(
            params={
                "data_path": "data/Salary_dataset.csv",
                "target_column": "Salary",
                "latency_budget": 120.0,
                "memory_limit_mb": 0.0,
            }
        )

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            current_stage = result.observation.stage

            if current_stage is None:
                break

            action = ModelSelectorAction(stage=current_stage)

            result = await env.step(action)

            reward = float(result.reward or 0.0)
            done = bool(result.done)

            metadata = {}
            if hasattr(result.observation, "metadata") and result.observation.metadata:
                metadata = result.observation.metadata

            error = metadata.get("error")

            reports = metadata.get("reports", {})
            current_report = reports.get(current_stage, {})

            rewards.append(reward)
            steps_taken = step

            step_outputs.append(
                {
                    "step": step,
                    "stage": current_stage,
                    "reward": reward,
                    "done": done,
                    "next_stage": getattr(result.observation, "stage", None),
                    "progress": getattr(result.observation, "progress", 0.0),
                    "dataset_profile": getattr(result.observation, "dataset_profile", {}),
                    "partial_pipeline": getattr(result.observation, "partial_pipeline", {}),
                    "report": current_report,
                }
            )

            log_step(
                step=step,
                action=current_stage,
                reward=reward,
                done=done,
                error=error,
            )

            if done:
                break

        if rewards:
            normalized_rewards = [max(0.0, min(1.0, r)) for r in rewards]
            score = sum(normalized_rewards) / len(normalized_rewards)
        else:
            score = 0.0

        score = max(0.0, min(1.0, score))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] inference failed: {exc}", flush=True)

    finally:
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )

        # Print ONLY the model_select result as requested
        model_result = next((s for s in step_outputs if s["stage"] == "model_select"), None)
        
        # Robustly find the report data
        final_report = None
        if model_result:
            final_report = model_result.get("report") or model_result.get("partial_pipeline", {}).get("model_select")
            
        if final_report:
            print("\n" + "=" * 45, flush=True)
            print("🚀 MODEL SELECTION FINAL RESULT", flush=True)
            print("=" * 45, flush=True)
            print(f"Selected Model : {final_report.get('selected_model', 'N/A')}", flush=True)
            
            # Extract score (it might be in 'score' or 'best_score')
            score_val = final_report.get('score') or final_report.get('best_score', 0.0)
            print(f"Base Score     : {float(score_val):.4f}", flush=True)
            
            print(f"Pipeline Stage : {model_result['stage']}", flush=True)
            print("=" * 45 + "\n", flush=True)
            
            # Wait a split second to ensure print buffer is drained on Windows
            sys.stdout.flush()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, RuntimeError):
        # Suppress noise during Windows event loop cleanup
        pass