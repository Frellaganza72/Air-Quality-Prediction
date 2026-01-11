#!/bin/bash

# Function to handle script exit
cleanup() {
    echo -e "\n🛑 Stopping all services..."
    # Kill all child processes in the current process group
    kill 0
    exit
}

# Trap signals for cleanup
trap cleanup SIGINT SIGTERM EXIT

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"

echo "==================================================="
echo "   Air Quality Monitoring System - Run Script"
echo "==================================================="

# Activate conda environment
echo "🔧 Activating conda environment 'airq'..."
eval "$(conda shell.bash hook)"
conda activate airq
if [ $? -ne 0 ]; then
    echo "❌ Failed to activate conda environment 'airq'"
    exit 1
fi
echo "✅ Conda environment 'airq' activated"

# Check for Python dependencies
echo "🔍 Checking Python dependencies..."
echo "📦 Installing requirements from backend/requirements.txt..."
# Clear pip cache and upgrade pip/setuptools first
pip3 install --upgrade pip setuptools wheel --quiet
pip3 install -r "$PROJECT_ROOT/backend/requirements.txt" --no-cache-dir --retries 5
if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies. Please check your python environment."
    echo "💡 Trying alternative installation method..."
    pip3 install -r "$PROJECT_ROOT/backend/requirements.txt" --retries 5
    if [ $? -ne 0 ]; then
        exit 1
    fi
fi
echo "✅ Dependencies checked and installed"

# Start Backend
echo "🚀 Starting Backend (Port 2000)..."
cd "$PROJECT_ROOT/backend"
python3 app.py &
BACKEND_PID=$!

# Wait a moment for backend to initialize
sleep 3

# Start Frontend
echo "🚀 Starting Frontend..."
cd "$PROJECT_ROOT/frontend-react"
# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    npm install
fi
npm run dev &
FRONTEND_PID=$!

echo "==================================================="
echo "   ✅ App is running!"
echo "   🖥️  Frontend: http://localhost:5173 (check output for exact URL)"
echo "   ⚙️  Backend:  http://localhost:2000"
echo "   Press Ctrl+C to stop everything."
echo "==================================================="

# Wait for processes
wait
