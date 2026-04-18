@echo off
if not exist venv (
    python -m venv venv
)

call venv\Scripts\activate
pip install -r requirements.txt

uvicorn app.routes:app --reload --host 0.0.0.0 --port 8000
pause