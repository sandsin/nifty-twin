#!/bin/sh
# Start Nifty Twin in the background on http://localhost:8501 (logs in nifty-twin.log).
cd "$(dirname "$0")"
pkill -f "streamlit run app.py" 2>/dev/null
nohup .venv/bin/streamlit run app.py > nifty-twin.log 2>&1 &
echo "Nifty Twin running at http://localhost:8501 (pid $!). Stop with: pkill -f 'streamlit run app.py'"
