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
    from models import ModelSelectorAction, ModelSelectorObservation, EnvInput
    from server.model_selector_environment import ModelSelectorEnvironment
except (ImportError, ValueError):
    try:
        # Fallback to local import structure
        from .model_selector_environment import ModelSelectorEnvironment
        from ..models import ModelSelectorAction, ModelSelectorObservation, EnvInput
    except (ImportError, ValueError):
        # Final fallback for flat local run from within server/
        from model_selector_environment import ModelSelectorEnvironment
        from models import ModelSelectorAction, ModelSelectorObservation, EnvInput


# Create a singleton environment so state persists across requests
_shared_env = ModelSelectorEnvironment()

def get_shared_env():
    return _shared_env


# Create the app with web interface and README integration
app = create_app(
    get_shared_env,
    ModelSelectorAction,
    ModelSelectorObservation,
    env_name="model_selector",
    max_concurrent_envs=1,
)

from fastapi.responses import HTMLResponse

@app.get("/reset-form", response_class=HTMLResponse, include_in_schema=False)
def reset_form_ui():
    """Full environment control UI - Reset, Step, and State in one page."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RL-Driven AutoML - Environment Control</title>
    <style>
        body { font-family: system-ui, sans-serif; max-width: 780px; margin: 48px auto; padding: 0 20px; color: #111; }
        h1 { font-size: 1.4rem; font-weight: 700; margin-bottom: 4px; }
        p.sub { color: #666; font-size: 0.9rem; margin-bottom: 32px; }
        h2 { font-size: 1rem; font-weight: 700; margin: 0 0 16px; padding-bottom: 8px;
             border-bottom: 2px solid #4f46e5; color: #4f46e5; }
        .card { border: 1px solid #e5e7eb; border-radius: 10px; padding: 20px 24px; margin-bottom: 20px; }
        label { display: block; font-size: 0.82rem; font-weight: 600; margin-bottom: 4px; color: #555; }
        input { width: 100%; padding: 8px 10px; border: 1px solid #d1d5db; border-radius: 6px;
                font-size: 0.92rem; box-sizing: border-box; margin-bottom: 14px; }
        input:focus { outline: 2px solid #4f46e5; border-color: #4f46e5; }
        .hint { font-size: 0.75rem; color: #9ca3af; margin-top: -11px; margin-bottom: 12px; }
        .btn-row { display: flex; gap: 10px; margin-top: 4px; }
        button { background: #4f46e5; color: white; border: none; padding: 9px 20px;
                 border-radius: 6px; font-size: 0.9rem; font-weight: 600; cursor: pointer; flex: 1; }
        button:hover { background: #4338ca; }
        button.secondary { background: #6b7280; }
        button.secondary:hover { background: #4b5563; }
        pre { background: #f8f9fa; border: 1px solid #e5e7eb; padding: 16px; border-radius: 8px;
              font-size: 0.82rem; white-space: pre-wrap; word-break: break-all;
              min-height: 100px; margin: 0; color: #1a1a1a; }
        hr { border: none; border-top: 1px solid #e5e7eb; margin: 28px 0; }
        .links { font-size: 0.82rem; color: #6b7280; }
        .links a { color: #4f46e5; text-decoration: none; margin-right: 16px; }
        .section-label { font-size: 0.78rem; font-weight: 700; color: #6b7280; text-transform: uppercase;
                         letter-spacing: 0.05em; margin-bottom: 8px; }
    </style>
</head>
<body>
    <h1>RL-Driven AutoML</h1>
    <p class="sub">Interact with the AutoML pipeline environment — Reset, Step through stages, and inspect State.</p>

    <!-- RESET -->
    <div class="card">
        <h2>1. Reset Environment</h2>
        <label for="data_path">Data Path <span style="color:red">*</span></label>
        <input type="text" id="data_path" placeholder="e.g. data/Salary_dataset.csv" />

        <label for="target_column">Target Column</label>
        <input type="text" id="target_column" placeholder="e.g. Salary (blank = auto-detect)" />

        <label for="latency_budget">Latency Budget (seconds)</label>
        <input type="number" id="latency_budget" value="120" step="1" />

        <label for="memory_limit_mb">Memory Limit (MB)</label>
        <input type="number" id="memory_limit_mb" value="0" step="1" />
        <p class="hint">0 = no limit</p>

        <label for="goal">Goal (optional)</label>
        <input type="text" id="goal" placeholder="e.g. minimize_latency" />

        <label for="metric">Metric (optional)</label>
        <input type="text" id="metric" placeholder="e.g. accuracy, rmse" />

        <div class="btn-row">
            <button onclick="doReset()">Reset Environment</button>
        </div>
    </div>

    <!-- STEP -->
    <div class="card">
        <h2>2. Execute Step</h2>
        <label for="stage">Stage Name <span style="color:red">*</span></label>
        <input type="text" id="stage" placeholder="e.g. cleaning, encoding, engineering, scaling, selection, model_select, tuning, ensemble" />
        <p class="hint">Must match the current expected stage from the Reset observation.</p>
        <div class="btn-row">
            <button onclick="doStep()">Send Step</button>
        </div>
    </div>

    <!-- STATE -->
    <div class="card">
        <h2>3. Get Current State</h2>
        <p style="font-size:0.88rem; color:#555; margin: 0 0 14px;">
            Fetch the current environment state including episode id, step count, current stage, and progress.
        </p>
        <div class="btn-row">
            <button onclick="doGetState()">Get State</button>
            <button class="secondary" onclick="clearOutput()">Clear</button>
        </div>
    </div>

    <!-- RESPONSE -->
    <div class="card">
        <div class="section-label">API Response</div>
        <pre id="output">// Responses will appear here...</pre>
    </div>

    <hr>
    <div class="links">
        <a href="/docs">Swagger API Docs</a>
        <a href="/state">Raw State JSON</a>
        <a href="/schema">View Schema</a>
    </div>

    <script>
    function setOutput(data) {
        document.getElementById('output').textContent =
            typeof data === 'string' ? data : JSON.stringify(data, null, 2);
    }
    function clearOutput() { setOutput('// Cleared.'); }

    async function doReset() {
        setOutput('// Resetting...');
        const params = {};
        [['data_path','text'],['target_column','text'],['latency_budget','number'],
         ['memory_limit_mb','number'],['goal','text'],['metric','text']].forEach(([f, t]) => {
            const v = document.getElementById(f).value;
            if (v) params[f] = t === 'number' ? parseFloat(v) : v;
        });
        try {
            const r = await fetch('/reset', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ params })
            });
            setOutput(await r.json());
        } catch(e) { setOutput('Error: ' + e.message); }
    }

    async function doStep() {
        const stage = document.getElementById('stage').value.trim();
        if (!stage) { setOutput('Error: Stage name is required.'); return; }
        setOutput('// Stepping: ' + stage + '...');
        try {
            const r = await fetch('/step', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ action: { stage } })
            });
            setOutput(await r.json());
        } catch(e) { setOutput('Error: ' + e.message); }
    }

    async function doGetState() {
        setOutput('// Fetching state...');
        try {
            const r = await fetch('/state');
            setOutput(await r.json());
        } catch(e) { setOutput('Error: ' + e.message); }
    }
    </script>
</body>
</html>"""


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