"""Canonical backend entrypoint.

This re-exports the Modal/FastAPI app from `App.py` so there is one obvious
module to target for local work and deployment commands.
"""

from App import ModelService, app, fastapi_app, model_service, web_app

__all__ = ["app", "fastapi_app", "web_app", "model_service", "ModelService"]
