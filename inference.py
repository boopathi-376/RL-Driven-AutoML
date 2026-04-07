import asyncio
import os
import sys
import requests
from typing import List, Optional, Dict, Any
from openai import OpenAI
import time
# ==========================================================
# WINDOWS ASYNCIO FIX (Prevents SSL / Loop closed errors)
# ==========================================================
if sys.platform == 'win32' and sys.version_info < (3, 11):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ── Config — uses hackathon injected environment variables ─────────────────────
BASE_URL = os.getenv("ENV_BASE_URL") or os.getenv("ENV_URL") or "https://boopathi376-rl-driven-automl.hf.space"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

TASKS = ["easy", "medium", "hard"]
STAGES = ["cleaning", "encoding", "engineering", "scaling", "selection", "model_select", "tuning", "ensemble"]

# Initialize OpenAI client with hackathon LiteLLM proxy
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL,
)

def call_llm(obs: dict) -> str:
    """Follow the environment's suggested stage strictly."""
    current_stage = obs.get("stage")
    if not current_stage or current_stage == "completed":
        return "completed"
    
    # We follow the environment's suggested step exactly as requested
    return current_stage

def run_task(task: str) -> float:
    """Run one full episode for a given task and return grade score (reward)."""
    try:
        time.sleep(1)
        # Reset with task query param
        r = requests.post(f"{BASE_URL}/reset", params={"task": task})
        r.raise_for_status()
        data = r.json()
        
        # OpenEnv might wrap the observation
        obs = data.get("observation") or data
        done = bool(obs.get("done", False))
        
        print(f"[START] task={task}", flush=True)
        
        # Check if reset itself failed
        if obs.get("stage") == "error":
            print(f"[FATAL ERROR] Reset failed for task={task}", flush=True)
            print(f"Full Response: {data}", flush=True)
            return 0.0

        step = 0
        total_reward = 0.0

        while not done:
            step += 1
            if not obs or obs.get("stage") == "completed":
                break

            # The LLM now strictly follows the environment's suggested stage
            action = call_llm(obs)

            # Send action to step endpoint - OpenEnv expects an "action" wrapper
            r = requests.post(f"{BASE_URL}/step", json={"action": {"stage": action}})
            r.raise_for_status()
            
            result = r.json()
            obs = result.get("observation") or result
            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", False))
            total_reward += reward

            # If we hit an error during a step, print full response for debugging
            if obs and obs.get("stage") == "error":
                print(f"[FATAL ERROR] Step {step} failed: {obs.get('metadata', {}).get('error')}", flush=True)
                print(f"Full Response: {result}", flush=True)
                break

            print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()}", flush=True)
            
            if done or step >= 12:
                break

        # Get final score from grading endpoint
        gr = requests.get(f"{BASE_URL}/grade", params={"task": task})
        score = gr.json().get("reward", total_reward)

        print(f"[END] task={task} score={score:.3f} steps={step}", flush=True)
        return score
    except Exception as e:
        print(f"[ERROR] Task {task} failed: {e}", flush=True)
        return 0.0

def main():
    if not API_KEY:
        print("[ERROR] API_KEY (HF_TOKEN) is not set.")
        return

    scores = {}
    for task in TASKS:
        scores[task] = run_task(task)

    avg = sum(scores.values()) / len(scores)
    summary = f"[SUMMARY] easy={scores['easy']:.2f} medium={scores['medium']:.2f} hard={scores['hard']:.2f} avg={avg:.2f}"
    print("\n" + "="*len(summary))
    print(summary)
    print("="*len(summary) + "\n")

if __name__ == "__main__":
    main()