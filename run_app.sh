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

# Check for Python dependencies
echo "🔍 Checking Python dependencies..."
python3 "$PROJECT_ROOT/backend/check_deps.py"
if [ $? -ne 0 ]; then
    echo "⚠️  Dependencies missing or incomplete."
    echo "📦 Installing requirements from backend/requirements.txt..."
    pip3 install -r "$PROJECT_ROOT/backend/requirements.txt"
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies. Please check your python environment."
        exit 1
    fi
    
    # Re-check to confirm
    python3 "$PROJECT_ROOT/backend/check_deps.py"
    if [ $? -ne 0 ]; then
        echo "❌ Still missing dependencies after installation. Please check manually."
        exit 1
    fi
fi

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
