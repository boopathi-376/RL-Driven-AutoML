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

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    # Try absolute import (best for Docker/Production with PYTHONPATH=/app)
    from models import ModelSelectorAction, ModelSelectorObservation
    from server.model_selector_environment import ModelSelectorEnvironment
except (ImportError, ValueError):
    try:
        # Fallback to local import structure
        from .model_selector_environment import ModelSelectorEnvironment
        from ..models import ModelSelectorAction, ModelSelectorObservation
    except (ImportError, ValueError):
        # Final fallback for flat local run from within server/
        from model_selector_environment import ModelSelectorEnvironment
        from models import ModelSelectorAction, ModelSelectorObservation

from fastapi.responses import HTMLResponse, RedirectResponse

# Create the app with web interface and README integration
app = create_app(
    ModelSelectorEnvironment,
    ModelSelectorAction,
    ModelSelectorObservation,
    env_name="model_selector",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)

@app.get("/", response_class=HTMLResponse)
def home():
    return custom_playground()

@app.get("/playground", response_class=HTMLResponse)
def custom_playground():
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
            }
            .badge {
                display: inline-block;
                padding: 4px 12px;
                background: rgba(16, 185, 129, 0.1);
                color: var(--accent);
                border-radius: 99px;
                font-size: 0.8rem;
                margin-bottom: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="badge">OpenEnv v0.2.2</div>
            <h1>Automl Pipeline Playground</h1>
            <p class="desc">Interactive interface for manual pipeline orchestration.</p>

            <div class="grid">
                <div class="section">
                    <label>1. Reset Environment</label>
                    <textarea id="resetBody" rows="8">{
  "params": {
    "data_path": "data/Salary_dataset.csv",
    "target_column": "Salary",
    "latency_budget": 120.0
  }
}</textarea>
                    <div class="controls">
                        <button class="btn-primary" onclick="resetEnv()">Execute Reset</button>
                    </div>
                </div>

                <div class="section">
                    <label>2. Execute Step</label>
                    <textarea id="stepBody" rows="8">{
  "action": {
    "stage": "cleaning"
  }
}</textarea>
                    <div class="controls">
                        <button class="btn-primary" onclick="stepEnv()">Send Action</button>
                        <button class="btn-secondary" onclick="getState()">Refresh State</button>
                    </div>
                </div>
            </div>

            <label>Live API Response</label>
            <pre id="output">// Responses will appear here...</pre>
        </div>

        <script>
            async function resetEnv() {
                updateOutput("// Resetting...");
                try {
                    const response = await fetch("/reset", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: document.getElementById("resetBody").value
                    });
                    const data = await response.json();
                    updateOutput(data);
                } catch (e) { updateOutput({error: e.message}); }
            }

            async function stepEnv() {
                updateOutput("// Stepping...");
                try {
                    const response = await fetch("/step", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: document.getElementById("stepBody").value
                    });
                    const data = await response.json();
                    updateOutput(data);
                } catch (e) { updateOutput({error: e.message}); }
            }

            async function getState() {
                updateOutput("// Fetching state...");
                try {
                    const response = await fetch("/state");
                    const data = await response.json();
                    updateOutput(data);
                } catch (e) { updateOutput({error: e.message}); }
            }

            function updateOutput(data) {
                const el = document.getElementById("output");
                if (typeof data === 'string') el.textContent = data;
                else el.textContent = JSON.stringify(data, null, 2);
            }
        </script>
    </body>
    </html>
    """


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()