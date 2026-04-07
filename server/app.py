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

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def reset_form_ui():
    """Simple default Reset UI - form-based interface for the /reset endpoint."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RL-Driven AutoML - Reset Environment</title>
    <style>
        body { font-family: system-ui, sans-serif; max-width: 700px; margin: 60px auto; padding: 0 20px; color: #111; }
        h1 { font-size: 1.4rem; font-weight: 700; margin-bottom: 4px; }
        p.sub { color: #666; font-size: 0.9rem; margin-bottom: 28px; }
        label { display: block; font-size: 0.85rem; font-weight: 600; margin-bottom: 4px; color: #333; }
        input, select { width: 100%; padding: 8px 10px; border: 1px solid #ccc; border-radius: 6px;
                        font-size: 0.95rem; box-sizing: border-box; margin-bottom: 16px; }
        input:focus, select:focus { outline: 2px solid #4f46e5; border-color: #4f46e5; }
        button { background: #4f46e5; color: white; border: none; padding: 10px 24px;
                 border-radius: 6px; font-size: 0.95rem; font-weight: 600; cursor: pointer; width: 100%; }
        button:hover { background: #4338ca; }
        pre { background: #f4f4f5; padding: 16px; border-radius: 8px; font-size: 0.85rem;
              white-space: pre-wrap; word-break: break-all; min-height: 80px; margin-top: 24px; }
        .field-hint { font-size: 0.78rem; color: #888; margin-top: -12px; margin-bottom: 12px; }
        hr { border: none; border-top: 1px solid #e5e7eb; margin: 24px 0; }
        .links { font-size: 0.85rem; color: #666; }
        .links a { color: #4f46e5; text-decoration: none; margin-right: 16px; }
    </style>
</head>
<body>
    <h1>RL-Driven AutoML - Environment</h1>
    <p class="sub">Use this form to reset the pipeline environment with your dataset.</p>

    <label for="data_path">Data Path <span style="color:red">*</span></label>
    <input type="text" id="data_path" placeholder="e.g. data/Salary_dataset.csv" />

    <label for="target_column">Target Column</label>
    <input type="text" id="target_column" placeholder="e.g. Salary (leave blank for auto-detect)" />

    <label for="latency_budget">Latency Budget (seconds)</label>
    <input type="number" id="latency_budget" value="120" step="1" />

    <label for="memory_limit_mb">Memory Limit (MB)</label>
    <input type="number" id="memory_limit_mb" value="0" step="1" />
    <p class="field-hint">Set to 0 for no limit.</p>

    <label for="goal">Goal</label>
    <input type="text" id="goal" placeholder="e.g. minimize_latency (optional)" />

    <label for="metric">Metric</label>
    <input type="text" id="metric" placeholder="e.g. accuracy, rmse (optional)" />

    <button onclick="doReset()">Reset Environment</button>

    <pre id="output">// Response will appear here after reset...</pre>

    <hr>
    <div class="links">
        <a href="/docs">Swagger API Docs</a>
        <a href="/state">View Current State</a>
        <a href="/schema">View Schema</a>
    </div>

    <script>
    async function doReset() {
        const btn = document.querySelector('button');
        btn.textContent = 'Resetting...';
        btn.disabled = true;

        const params = {};
        const fields = ['data_path','target_column','latency_budget','memory_limit_mb','goal','metric'];
        fields.forEach(f => {
            const el = document.getElementById(f);
            if (el.value !== '' && el.value !== '0' || f === 'data_path') {
                if (el.type === 'number') params[f] = parseFloat(el.value);
                else if (el.value) params[f] = el.value;
            }
        });

        try {
            const resp = await fetch('/reset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ params })
            });
            const data = await resp.json();
            document.getElementById('output').textContent = JSON.stringify(data, null, 2);
        } catch(e) {
            document.getElementById('output').textContent = 'Error: ' + e.message;
        }

        btn.textContent = 'Reset Environment';
        btn.disabled = false;
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