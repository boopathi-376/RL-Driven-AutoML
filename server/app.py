# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Model Selector Environment.

This module creates an HTTP server that exposes the ModelSelectorEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions
    - GET /playground: Interactive AutoML dashboard

Usage:
    # Development (with auto-reload):
    uvrun server --reload

    # Or run directly:
    python -m server.app
"""

import sys
from pathlib import Path
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi import HTTPException
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================================
# ROBUST PATH HANDLING (Fixes Docker/Production Imports)
# ==========================================================
# Add project root to sys.path to allow absolute imports
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as e:
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

# Standard imports
from models import ModelSelectorAction, ModelSelectorObservation
try:
    from server.model_selector_environment import ModelSelectorEnvironment
except ImportError:
    from model_selector_environment import ModelSelectorEnvironment


# ==========================================================
# SHARED ENVIRONMENT (Stateful REST)
# ==========================================================
# Create a singleton wrapper so HTTP REST calls (Swagger/Playground) 
# maintain state sequentially in memory.
_shared_env = ModelSelectorEnvironment()

def get_shared_env():
    return _shared_env

app = create_app(
    get_shared_env,
    ModelSelectorAction,
    ModelSelectorObservation,
    env_name="model_selector",
    max_concurrent_envs=1,
)

# ==========================================================
# CUSTOM UI ENDPOINTS
# ==========================================================

@app.get("/", include_in_schema=False)
def root_redirect():
    """Redirects the root URL to the interactive playground."""
    return RedirectResponse(url="/playground")

@app.get("/playground", response_class=HTMLResponse)
def custom_playground():
    """Returns a premium, dark-themed interactive AutoML playground."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AutoML Pipeline Playground</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&family=JetBrains+Mono&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary: #8b5cf6;
                --secondary: #3b82f6;
                --bg: #0f172a;
                --card: #1e293b;
                --text: #f8fafc;
                --accent: #10b981;
                --error: #ef4444;
            }
            body {
                font-family: 'Outfit', sans-serif;
                background: var(--bg);
                color: var(--text);
                margin: 0;
                padding: 40px;
                display: flex;
                flex-direction: column;
                align-items: center;
                min-height: 100vh;
            }
            .container {
                width: 100%;
                max-width: 1000px;
                background: var(--card);
                padding: 40px;
                border-radius: 24px;
                box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
                border: 1px solid rgba(255,255,255,0.1);
            }
            h1 {
                font-weight: 600;
                background: linear-gradient(to right, #a78bfa, #60a5fa);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 8px;
            }
            p.desc { color: #94a3b8; margin-bottom: 32px; }
            .grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 24px;
                margin-bottom: 32px;
            }
            .section {
                background: rgba(15, 23, 42, 0.5);
                padding: 20px;
                border-radius: 16px;
                border: 1px solid rgba(255,255,255,0.05);
            }
            label {
                display: block;
                font-weight: 600;
                margin-bottom: 12px;
                color: #cbd5e1;
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
            textarea {
                width: 100%;
                background: #0f172a;
                color: #38bdf8;
                border: 1px solid #334155;
                border-radius: 12px;
                padding: 16px;
                font-family: 'JetBrains Mono', monospace;
                font-size: 13px;
                resize: vertical;
                box-sizing: border-box;
                transition: border-color 0.2s;
            }
            textarea:focus {
                outline: none;
                border-color: var(--primary);
            }
            .controls {
                display: flex;
                gap: 12px;
                margin-top: 16px;
            }
            button {
                flex: 1;
                padding: 12px;
                border-radius: 12px;
                border: none;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s;
                font-family: 'Outfit', sans-serif;
            }
            .btn-primary { background: linear-gradient(135deg, var(--primary), var(--secondary)); color: white; }
            .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 10px 15px -3px rgba(139, 92, 246, 0.3); }
            .btn-secondary { background: #334155; color: #f1f5f9; }
            .btn-secondary:hover { background: #475569; }
            .btn-error { background: var(--error); color: white; }
            
            #output {
                width: 100%;
                background: #020617;
                padding: 24px;
                border-radius: 16px;
                font-family: 'JetBrains Mono', monospace;
                font-size: 13px;
                color: #10b981;
                overflow-x: auto;
                border: 1px solid #1e293b;
                max-height: 300px;
                overflow-y: auto;
            }
            .state-dashboard {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 16px;
                margin-bottom: 24px;
            }
            .stat-card {
                background: rgba(139, 92, 246, 0.1);
                border: 1px solid rgba(139, 92, 246, 0.2);
                padding: 16px;
                border-radius: 16px;
                text-align: center;
                transition: all 0.2s;
            }
            .stat-card:hover {
                background: rgba(139, 92, 246, 0.15);
                transform: translateY(-2px);
            }
            .stat-value {
                font-size: 1.2rem;
                font-weight: 600;
                color: var(--primary);
                display: block;
                margin-bottom: 8px;
            }
            .stat-label {
                font-size: 0.7rem;
                text-transform: uppercase;
                color: #94a3b8;
                letter-spacing: 0.1em;
            }
            .guide {
                margin-top: 32px;
                padding-top: 32px;
                border-top: 1px solid #1e293b;
            }
            .guide h3 { font-size: 1rem; color: #cbd5e1; margin-bottom: 16px; }
            .guide ul { padding-left: 20px; color: #94a3b8; font-size: 0.9rem; line-height: 1.6; }
            .badge {
                display: inline-block;
                padding: 4px 12px;
                background: rgba(16, 185, 129, 0.1);
                color: var(--accent);
                border-radius: 99px;
                font-size: 0.8rem;
                margin-bottom: 20px;
            }
            .loading {
                opacity: 0.7;
                pointer-events: none;
            }
            .error-text {
                color: var(--error);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="badge">OpenEnv v0.2.2</div>
            <h1>🤖 AutoML Pipeline Playground</h1>
            <p class="desc">Interactive environment for automated machine learning pipeline selection</p>
            
            <!-- State Dashboard -->
            <div class="state-dashboard">
                <div class="stat-card">
                    <span class="stat-label">Current Stage</span>
                    <span id="stat-current-stage" class="stat-value">-</span>
                </div>
                <div class="stat-card">
                    <span class="stat-label">Next Stage</span>
                    <span id="stat-next-stage" class="stat-value">-</span>
                </div>
                <div class="stat-card">
                    <span class="stat-label">Step Count</span>
                    <span id="stat-current-step" class="stat-value">0</span>
                </div>
                <div class="stat-card">
                    <span class="stat-label">Latest Reward</span>
                    <span id="stat-reward" class="stat-value">0.000</span>
                </div>
            </div>

            <div class="grid">
                <div class="section">
                    <label>📊 1. Reset Environment</label>
                    <textarea id="resetBody" rows="7">{
  "params": {
    "data_path": "data/Salary_dataset.csv",
    "target_column": "Salary",
    "latency_budget": 120.0
  }
}</textarea>
                    <div class="controls">
                        <button class="btn-primary" onclick="resetEnv()">🔄 Execute Reset</button>
                    </div>
                </div>

                <div class="section">
                    <label>⚡ 2. Execute Step</label>
                    <textarea id="stepBody" rows="7">{
  "action": {
    "stage": "cleaning"
  }
}</textarea>
                    <div class="controls">
                        <button class="btn-primary" onclick="stepEnv()">🚀 Send Action</button>
                        <button class="btn-secondary" onclick="getState()">📡 Refresh State</button>
                    </div>
                </div>
            </div>

            <label>📝 Live API Response</label>
            <pre id="output">// Click "Execute Reset" to start the AutoML pipeline...</pre>

            <!-- Final Report (Hidden until Done) -->
            <div id="final-report" style="display: none; margin-top: 24px; padding: 24px; background: rgba(16, 185, 129, 0.1); border: 2px solid var(--accent); border-radius: 16px;">
                <h2 style="margin-top: 0; color: var(--accent); font-size: 1.2rem; display: flex; align-items: center; gap: 8px;">
                    🏆 Pipeline Success: Final Report
                </h2>
                <div id="final-stats" style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                    <!-- Populated by JS -->
                </div>
            </div>

            <!-- Operation Guide -->
            <div class="guide">
                <h3>📖 Pipeline Documentation</h3>
                <ul>
                    <li><strong>Step 1 - Reset:</strong> Initialize the environment with your dataset configuration</li>
                    <li><strong>Step 2 - Follow Stages:</strong> The environment will guide you through the pipeline stages</li>
                    <li><strong>Dataset Types:</strong> For <b>.txt</b> files, <code>target_column</code> is not required (unsupervised)</li>
                    <li><strong>Pipeline Order:</strong> cleaning → encoding → engineering → scaling → selection → modeling → tuning → ensemble</li>
                    <li><strong>Reward System:</strong> Positive rewards for progress, negative for invalid actions</li>
                    <li><strong>Termination:</strong> Environment ends when pipeline completes or budget exhausted</li>
                </ul>
            </div>
        </div>

        <script>
            // Helper function to update dashboard with observation data
            function updateDashboard(data) {
                // Extract observation from response
                const observation = data.observation || data;
                
                // 1. Update Current Stage
                const currentStage = observation.current_stage || observation.stage;
                const stageElement = document.getElementById('stat-current-stage');
                if (stageElement) {
                    stageElement.textContent = currentStage || '-';
                }

                // 2. Update Next Stage
                const nextStage = observation.next_stage || observation.next_stage_name;
                const nextStageElement = document.getElementById('stat-next-stage');
                if (nextStageElement) {
                    if (nextStage && nextStage !== 'complete') {
                        nextStageElement.textContent = nextStage;
                    } else if (observation.done || data.done) {
                        nextStageElement.textContent = '🏁 Done';
                    } else {
                        nextStageElement.textContent = '-';
                    }
                }
                
                // 3. Update Current Step
                const currentStep = observation.current_step ?? observation.step_count ?? data.step_count;
                const stepElement = document.getElementById('stat-current-step');
                if (stepElement && currentStep !== undefined && currentStep !== null) {
                    stepElement.textContent = currentStep;
                }
                
                // 4. Update Reward
                const reward = observation.reward ?? data.reward;
                const rewardElement = document.getElementById('stat-reward');
                if (rewardElement && reward !== undefined && reward !== null) {
                    rewardElement.textContent = typeof reward === 'number' ? reward.toFixed(3) : reward;
                    if (reward > 0) {
                        rewardElement.style.color = '#10b981';
                    } else if (reward < 0) {
                        rewardElement.style.color = '#ef4444';
                    } else {
                        rewardElement.style.color = '#8b5cf6';
                    }
                }

                // 5. Final Report Logic
                const reportEl = document.getElementById('final-report');
                if (observation.done || data.done) {
                    reportEl.style.display = 'block';
                    const statsEl = document.getElementById('final-stats');
                    
                    const pipeline = observation.partial_pipeline || {};
                    const modelData = pipeline.model_select || {};
                    const tuningData = pipeline.tuning || {};
                    
                    statsEl.innerHTML = `
                        <div style="background: rgba(0,0,0,0.2); padding: 12px; border-radius: 8px;">
                            <span style="color: #94a3b8; font-size: 0.7rem; text-transform: uppercase;">Selected Model</span>
                            <div style="font-weight: 600; font-size: 1.1rem; color: #fff;">${modelData.selected_model || 'N/A'}</div>
                        </div>
                        <div style="background: rgba(0,0,0,0.2); padding: 12px; border-radius: 8px;">
                            <span style="color: #94a3b8; font-size: 0.7rem; text-transform: uppercase;">Final Score</span>
                            <div style="font-weight: 600; font-size: 1.1rem; color: #10b981;">${(modelData.score || tuningData.best_score || 0.0).toFixed(4)}</div>
                        </div>
                    `;
                } else {
                    reportEl.style.display = 'none';
                }
            }

            async function resetEnv() {
                const outputEl = document.getElementById('output');
                outputEl.textContent = "🔄 Resetting environment...";
                document.getElementById('final-report').style.display = 'none';
                
                try {
                    const resetData = JSON.parse(document.getElementById("resetBody").value);
                    const response = await fetch("/reset", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(resetData)
                    });
                    
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }
                    
                    const data = await response.json();
                    outputEl.textContent = JSON.stringify(data, null, 2);
                    updateDashboard(data);
                    
                    // AUTO-UPDATE NEXT ACTION
                    const obs = data.observation || data;
                    if (obs.stage) {
                        const stepBody = { action: { stage: obs.stage } };
                        document.getElementById("stepBody").value = JSON.stringify(stepBody, null, 2);
                    }
                    
                } catch (e) {
                    outputEl.textContent = `❌ Error: ${e.message}`;
                }
            }
            
            async function stepEnv() {
                const outputEl = document.getElementById('output');
                outputEl.textContent = "⚡ Executing action...";
                
                try {
                    const stepData = JSON.parse(document.getElementById("stepBody").value);
                    const response = await fetch("/step", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(stepData)
                    });
                    
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }
                    
                    const data = await response.json();
                    outputEl.textContent = JSON.stringify(data, null, 2);
                    updateDashboard(data);
                    
                    // AUTO-UPDATE NEXT ACTION
                    const obs = data.observation || data;
                    if (obs.stage && obs.stage !== 'completed') {
                        const stepBody = { action: { stage: obs.stage } };
                        document.getElementById("stepBody").value = JSON.stringify(stepBody, null, 2);
                    }
                    
                } catch (e) {
                    outputEl.textContent = `❌ Error: ${e.message}`;
                }
            }
            
            async function getState() {
                const outputEl = document.getElementById('output');
                outputEl.textContent = "📡 Fetching current state...";
                
                try {
                    const response = await fetch("/state");
                    
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }
                    
                    const data = await response.json();
                    outputEl.textContent = JSON.stringify(data, null, 2);
                    updateDashboard(data);
                    
                } catch (e) {
                    outputEl.textContent = `❌ Error: ${e.message}`;
                    console.error('Get state error:', e);
                }
            }
            
            // Auto-refresh state every 5 seconds (optional)
            let autoRefresh = false;
            let refreshInterval;
            
            function toggleAutoRefresh() {
                autoRefresh = !autoRefresh;
                if (autoRefresh) {
                    refreshInterval = setInterval(getState, 5000);
                    console.log('Auto-refresh enabled');
                } else {
                    if (refreshInterval) {
                        clearInterval(refreshInterval);
                        console.log('Auto-refresh disabled');
                    }
                }
            }
            
            // Load initial state on page load
            window.addEventListener('load', () => {
                getState();
            });
        </script>
    </body>
    </html>
    """


# ==========================================================
# CLI ENTRYPOINT
# ==========================================================

def main():
    """
    Entry point for server execution.
    
    Usage:
        uvrun server                   # default host=0.0.0.0, port=7860
        uvrun server --port 8001
        uvrun server --host 127.0.0.1 --port 8080 --reload
    """
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Model Selector Environment Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Bind port (default: 7860)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    args = parser.parse_args()

    logger.info(f"Starting Model Selector Environment Server on http://{args.host}:{args.port}")
    logger.info(f"Playground available at http://{args.host}:{args.port}/playground")
    
    uvicorn.run(
        "server.app:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()