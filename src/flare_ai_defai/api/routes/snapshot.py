import json
from pathlib import Path
from fastapi import APIRouter, HTTPException

from flare_ai_defai.settings import settings

router = APIRouter()

def _snapshot_path() -> Path:
    # settings.latest_update_path should be relative to repo root / container WORKDIR
    return Path(settings.latest_update_path).resolve()

@router.get("/snapshot")
def get_snapshot():
    p = _snapshot_path()
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Missing snapshot file: {p}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid JSON in {p}: {e}") from e

