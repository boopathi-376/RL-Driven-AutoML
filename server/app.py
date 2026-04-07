# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import sys
import time
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as e:
    raise ImportError("openenv is required. Install with 'uv sync'") from e

# Import logic
try:
    from models import ModelSelectorAction, ModelSelectorObservation
    from model_selector_environment import ModelSelectorEnvironment
except ImportError:
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from models import ModelSelectorAction, ModelSelectorObservation
    from server.model_selector_environment import ModelSelectorEnvironment

# ==========================================================
# SHARED ENVIRONMENT (Singleton)
# ==========================================================
_shared_env = ModelSelectorEnvironment()

def get_shared_env():
    return _shared_env

# Initialize the base app
app = create_app(
    get_shared_env,
    ModelSelectorAction,
    ModelSelectorObservation,
    env_name="model_selector",
    max_concurrent_envs=1
)

# ==========================================================
# FIX: Explicit State Endpoint 
# ==========================================================
@app.get("/state")
async def get_current_state():
    """Explicitly returns the current environment state to avoid stale data."""
    env = get_shared_env()
    # Ensure we return a dictionary compatible with your JS
    return {
        "observation": env.get_observation(),
        "step_count": getattr(env, "steps_taken", 0),
        "reward": getattr(env, "last_reward", 0.0),
        "timestamp": time.time() # Force unique response
    }

# ==========================================================
# UPDATED UI WITH CACHE-BUSTING JS
# ==========================================================

@app.get("/playground", response_class=HTMLResponse)
def custom_playground():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>AutoML Pipeline Playground</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&family=JetBrains+Mono&display=swap" rel="stylesheet">
        <style>
            :root { --primary: #8b5cf6; --secondary: #3b82f6; --bg: #0f172a; --card: #1e293b; --text: #f8fafc; --accent: #10b981; }
            body { font-family: 'Outfit', sans-serif; background: var(--bg); color: var(--text); padding: 40px; display: flex; flex-direction: column; align-items: center; }
            .container { width: 100%; max-width: 1000px; background: var(--card); padding: 40px; border-radius: 24px; border: 1px solid rgba(255,255,255,0.1); }
            .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 32px; }
            .section { background: rgba(15, 23, 42, 0.5); padding: 20px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.05); }
            textarea { width: 100%; background: #0f172a; color: #38bdf8; border: 1px solid #334155; border-radius: 12px; padding: 16px; font-family: 'JetBrains Mono'; font-size: 13px; box-sizing: border-box; }
            button { width: 100%; padding: 12px; border-radius: 12px; border: none; font-weight: 600; cursor: pointer; margin-top: 10px; }
            .btn-primary { background: linear-gradient(135deg, var(--primary), var(--secondary)); color: white; }
            .btn-secondary { background: #334155; color: #f1f5f9; }
            #output { background: #020617; padding: 24px; border-radius: 16px; font-family: 'JetBrains Mono'; color: #10b981; overflow-x: auto; border: 1px solid #1e293b; }
            .state-dashboard { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 24px; }
            .stat-card { background: rgba(139, 92, 246, 0.1); border: 1px solid rgba(139, 92, 246, 0.2); padding: 16px; border-radius: 16px; text-align: center; }
            .stat-value { font-size: 1.2rem; font-weight: 600; color: var(--primary); display: block; }
            .stat-label { font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>AutoML Pipeline Playground</h1>
            <div class="state-dashboard">
                <div class="stat-card">
                    <span class="stat-label">Current Stage</span>
                    <span id="stat-stage" class="stat-value">---</span>
                </div>
                <div class="stat-card">
                    <span class="stat-label">Steps Taken</span>
                    <span id="stat-steps" class="stat-value">0</span>
                </div>
                <div class="stat-card">
                    <span class="stat-label">Latest Reward</span>
                    <span id="stat-reward" class="stat-value">0.00</span>
                </div>
            </div>

            <div class="grid">
                <div class="section">
                    <label>1. Reset Environment</label>
                    <textarea id="resetBody" rows="5">{"params": {"data_path": "data/Salary_dataset.csv", "target_column": "Salary"}}</textarea>
                    <button class="btn-primary" onclick="executeAction('/reset', 'resetBody')">Reset</button>
                </div>
                <div class="section">
                    <label>2. Execute Step</label>
                    <textarea id="stepBody" rows="5">{"action": {"stage": "cleaning"}}</textarea>
                    <button class="btn-primary" onclick="executeAction('/step', 'stepBody')">Step</button>
                    <button class="btn-secondary" onclick="updateState()">Refresh State</button>
                </div>
            </div>

            <label>API Response</label>
            <pre id="output">// Ready...</pre>
        </div>

        <script>
            async function executeAction(endpoint, bodyId) {
                const out = document.getElementById("output");
                out.textContent = "// Processing...";
                try {
                    const res = await fetch(endpoint, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: document.getElementById(bodyId).value
                    });
                    const data = await res.json();
                    renderResponse(data);
                } catch (e) { out.textContent = "Error: " + e.message; }
            }

            async function updateState() {
                try {
                    // Use timestamp to bypass browser cache
                    const res = await fetch("/state?t=" + Date.now());
                    const data = await res.json();
                    renderResponse(data);
                } catch (e) { console.error(e); }
            }

            function renderResponse(data) {
                document.getElementById("output").textContent = JSON.stringify(data, null, 2);
                
                // Extract observation
                const obs = data.observation || data;
                
                // Map the keys precisely to your Environment's attribute names
                document.getElementById("stat-stage").textContent = obs.stage || obs.current_stage || "---";
                document.getElementById("stat-steps").textContent = data.step_count ?? obs.steps_taken ?? "0";
                document.getElementById("stat-reward").textContent = (data.reward ?? 0).toFixed(4);
            }
            
            // Auto-refresh on load
            window.onload = updateState;
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)