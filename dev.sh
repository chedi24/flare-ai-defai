#!/bin/bash
source .venv/bin/activate
pkill -f uvicorn || true
uvicorn flare_ai_defai.main:app --host 0.0.0.0 --port 8080 --reload
