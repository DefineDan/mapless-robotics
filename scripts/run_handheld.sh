#!/usr/bin/env bash
set -e
MAPLESS_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MAPLESS_PROJECT_ROOT"
export PYTHONPATH="$MAPLESS_PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -z "${DINOV3_WEIGHTS:-}" && -f "$MAPLESS_PROJECT_ROOT/external/dinov3/dinov3/models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth" ]]; then
    export DINOV3_WEIGHTS="$MAPLESS_PROJECT_ROOT/external/dinov3/dinov3/models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
fi
exec "$MAPLESS_PROJECT_ROOT/.venv/bin/python" -m mapless_robotics.handheld "$@"
