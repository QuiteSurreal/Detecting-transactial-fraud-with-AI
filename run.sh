#!/bin/bash
if [ ! -d "detectVenv" ]; then
    python3 -m detectVenv detectVenv
fi

source detectVenv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

uvicorn app.routes:app --reload --host 0.0.0.0 --port 8000