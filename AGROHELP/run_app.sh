#!/bin/bash

# Navigate to the project root directory
cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run the Streamlit app
python3 -m streamlit run app.py