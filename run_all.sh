#!/bin/bash
echo "🚀 Starting NIDS Project Automation Script..."

PROJECT_DIR="$HOME/NIDS_Project"

echo "📌 Step 1: Activating virtual environment..."
source $PROJECT_DIR/venv/bin/activate

echo "📌 Step 2: Running preprocess.py..."
python3 $PROJECT_DIR/src/preprocess.py

echo "📌 Step 3: Running train.py..."
python3 $PROJECT_DIR/src/train.py

echo "📌 Step 4: Running explain.py..."
python3 $PROJECT_DIR/src/explain.py

echo "📌 Step 5: Launching Streamlit Dashboard..."
streamlit run $PROJECT_DIR/app/app.py
