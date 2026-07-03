"""
WSGI entry point for production deployment (gunicorn, etc).

app.py lives inside Flask/, which is fine for `python app.py` locally,
but most PaaS platforms (Render, Railway, Heroku-style) run the start
command from the repo root. This file makes that work with a single,
simple command run from the repo root:

    gunicorn wsgi:app
"""
import os
import sys

FLASK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Flask")
if FLASK_DIR not in sys.path:
    sys.path.insert(0, FLASK_DIR)

from app import app  # noqa: E402  (import after sys.path setup, by design)
