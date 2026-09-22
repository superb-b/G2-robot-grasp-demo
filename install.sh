#!/usr/bin/env bash
#
# install.sh — build the virtualenv for G2_grape_235_v1 on a new machine.
#
# Run this ON THE TARGET MACHINE, in the project directory, after copying the
# sources across. Do NOT copy .venv from another machine: it is aarch64-specific
# and its interpreter paths are absolute, so it cannot be relocated.
#
# agibot_gdk is NOT installed here — it ships with the robot's system image and
# is only importable from the robot's own Python environment.
set -euo pipefail

cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
VENV=".venv"

if [[ ! -d wheelhouse ]]; then
    echo "error: wheelhouse/ not found in $(pwd)" >&2
    exit 1
fi
if [[ ! -f requirements.txt ]]; then
    echo "error: requirements.txt not found in $(pwd)" >&2
    exit 1
fi

echo "== interpreter =="
"$PY" -c 'import sys, platform; print(sys.version.split()[0], platform.machine())'

echo
echo "== creating $VENV =="
if [[ -d "$VENV" ]]; then
    echo "$VENV already exists — reusing it (delete it first for a clean build)"
else
    "$PY" -m venv "$VENV"
fi

echo
echo "== installing from wheelhouse (offline) =="
"$VENV/bin/pip" install --no-index --find-links wheelhouse --upgrade pip || true
"$VENV/bin/pip" install --no-index --find-links wheelhouse -r requirements.txt

echo
echo "== checking agibot_gdk =="
if "$VENV/bin/python" -c 'import agibot_gdk' 2>/dev/null; then
    echo "agibot_gdk importable"
else
    echo "agibot_gdk NOT importable here — expected unless this is the robot itself."
    echo "It comes from the robot's system image; run the pipeline with the robot's"
    echo "own python3 (not $VENV/bin/python) on the robot."
fi

source .venv/bin/activate
source ~/app/env.sh
source ~/app/gdk/scripts/env.sh

echo
echo "== done =="
echo "Run the pipeline with the robot's system python, e.g.:"
echo
echo "python3 grasp_pipeline.py --stream true --target bottle"

python3 grasp_pipeline.py --stream true