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
    reset_input_model=EnvInput,
    max_concurrent_envs=1,
)


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