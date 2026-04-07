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
    """3-column layout UI: Reset | Step | State"""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RL-Driven AutoML</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: system-ui, sans-serif; background: #f9fafb; color: #111; padding: 32px 24px; }
        header { margin-bottom: 28px; }
        header h1 { font-size: 1.35rem; font-weight: 700; }
        header p  { font-size: 0.87rem; color: #6b7280; margin-top: 4px; }

        /* 3-column grid */
        .columns { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 16px; }
        .col { background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 20px; }
        .col-title { font-size: 0.78rem; font-weight: 700; text-transform: uppercase;
                     letter-spacing: 0.06em; color: #4f46e5; margin-bottom: 14px;
                     padding-bottom: 8px; border-bottom: 2px solid #e0e7ff; }

        label { display: block; font-size: 0.78rem; font-weight: 600; color: #374151; margin-bottom: 3px; }
        input  { width: 100%; padding: 7px 9px; border: 1px solid #d1d5db; border-radius: 6px;
                 font-size: 0.85rem; margin-bottom: 10px; }
        input:focus { outline: 2px solid #4f46e5; border-color: #4f46e5; }
        .hint { font-size: 0.72rem; color: #9ca3af; margin-top: -8px; margin-bottom: 10px; }

        button { width: 100%; padding: 9px; border: none; border-radius: 7px;
                 font-size: 0.88rem; font-weight: 600; cursor: pointer;
                 background: #4f46e5; color: #fff; margin-top: 6px; }
        button:hover { background: #4338ca; }
        button.gray { background: #6b7280; margin-top: 8px; }
        button.gray:hover { background: #4b5563; }

        /* Response panel */
        .response-box { background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 20px; }
        .response-title { font-size: 0.78rem; font-weight: 700; text-transform: uppercase;
                          letter-spacing: 0.06em; color: #374151; margin-bottom: 10px; }
        pre { background: #f3f4f6; border-radius: 8px; padding: 14px; font-size: 0.8rem;
              white-space: pre-wrap; word-break: break-all; min-height: 120px; color: #1f2937; }

        footer { margin-top: 16px; font-size: 0.8rem; color: #9ca3af; }
        footer a { color: #4f46e5; text-decoration: none; margin-right: 14px; }

        /* state info list */
        .state-desc { font-size: 0.8rem; color: #6b7280; line-height: 1.6; margin-bottom: 14px; }
        .state-desc li { margin-left: 14px; }
    </style>
</head>
<body>
    <header>
        <h1>RL-Driven AutoML &mdash; Environment Control</h1>
        <p>Interact with the AutoML pipeline: Reset with your dataset, step through stages, and inspect state.</p>
    </header>

    <div class="columns">

        <!-- COL 1: RESET -->
        <div class="col">
            <div class="col-title">1 &mdash; Reset</div>

            <label for="data_path">Data Path <span style="color:#ef4444">*</span></label>
            <input type="text" id="data_path" placeholder="Salary_dataset.csv" />

            <label for="target_column">Target Column</label>
            <input type="text" id="target_column" placeholder="e.g. Salary" />

            <label for="latency_budget">Latency Budget (s)</label>
            <input type="number" id="latency_budget" value="120" step="1" />

            <label for="memory_limit_mb">Memory Limit (MB)</label>
            <input type="number" id="memory_limit_mb" value="0" step="1" />
            <p class="hint">0 = no limit</p>

            <label for="goal">Goal (optional)</label>
            <input type="text" id="goal" placeholder="e.g. minimize_latency" />

            <label for="metric">Metric (optional)</label>
            <input type="text" id="metric" placeholder="e.g. accuracy, rmse" />

            <button onclick="doReset()">Reset Environment</button>
        </div>

        <!-- COL 2: STEP -->
        <div class="col">
            <div class="col-title">2 &mdash; Step</div>
            <p class="state-desc">
                Send a pipeline stage action. The stage must match what the environment is currently expecting.
                <br><br>
                <b>8-stage (CSV):</b>
                <ul>
                    <li>cleaning</li>
                    <li>encoding</li>
                    <li>engineering</li>
                    <li>scaling</li>
                    <li>selection</li>
                    <li>model_select</li>
                    <li>tuning</li>
                    <li>ensemble</li>
                </ul>
                <br>
                <b>4-stage (TXT):</b>
                <ul>
                    <li>cleaning &rarr; model_select &rarr; tuning &rarr; ensemble</li>
                </ul>
            </p>
            <label for="stage">Stage Name <span style="color:#ef4444">*</span></label>
            <input type="text" id="stage" placeholder="e.g. cleaning" />
            <button onclick="doStep()">Send Step</button>
        </div>

        <!-- COL 3: STATE -->
        <div class="col">
            <div class="col-title">3 &mdash; State</div>
            <p class="state-desc">
                Inspect the current environment state at any time. Returns:
                <ul>
                    <li>episode_id</li>
                    <li>step_count</li>
                    <li>current_stage</li>
                    <li>next_stage</li>
                    <li>task_type</li>
                    <li>progress (0.0 &rarr; 1.0)</li>
                </ul>
            </p>
            <button onclick="doGetState()">Get State</button>
            <button class="gray" onclick="clearOutput()">Clear Response</button>
        </div>
    </div>

    <!-- RESPONSE PANEL -->
    <div class="response-box">
        <div class="response-title">API Response</div>
        <pre id="output">// Responses will appear here after actions...</pre>
    </div>

    <footer>
        <a href="/docs">Swagger Docs</a>
        <a href="/state">Raw State JSON</a>
        <a href="/schema">Schema</a>
    </footer>

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
         ['memory_limit_mb','number'],['goal','text'],['metric','text']].forEach(([f,t]) => {
            const v = document.getElementById(f).value;
            if (v) params[f] = t === 'number' ? parseFloat(v) : v;
        });
        try {
            const r = await fetch('/reset', {
                method:'POST', headers:{'Content-Type':'application/json'},
                body: JSON.stringify(params)
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
                method:'POST', headers:{'Content-Type':'application/json'},
                body: JSON.stringify({ stage })
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