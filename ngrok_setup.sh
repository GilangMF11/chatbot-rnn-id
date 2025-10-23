#!/bin/bash

# Script untuk menjalankan Streamlit dengan ngrok tunneling
# Pastikan ngrok sudah terinstall dan virtual environment aktif

echo "🤖 Chatbot RNN Bahasa Indonesia - Ngrok Setup"
echo "=============================================="

# Check if virtual environment is active
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️ Virtual environment not active!"
    echo "💡 Please activate 'chatbot_env' first:"
    echo "   source chatbot_env/bin/activate"
    exit 1
fi

# Check if ngrok is installed
if ! command -v ngrok &> /dev/null; then
    echo "❌ Ngrok not found!"
    echo "💡 Please install ngrok first:"
    echo "   - Download from: https://ngrok.com/download"
    echo "   - Or install via brew: brew install ngrok"
    echo "   - Or via pip: pip install pyngrok"
    exit 1
fi

# Check if model files exist
echo "🔍 Checking model files..."
missing_files=()
if [ ! -f "models/attention_rnn_model.h5" ]; then
    missing_files+=("models/attention_rnn_model.h5")
fi
if [ ! -f "models/attention_rnn_tokenizer.pkl" ]; then
    missing_files+=("models/attention_rnn_tokenizer.pkl")
fi
if [ ! -f "models/intent_classifier.pkl" ]; then
    missing_files+=("models/intent_classifier.pkl")
fi

if [ ${#missing_files[@]} -gt 0 ]; then
    echo "❌ Missing model files:"
    for file in "${missing_files[@]}"; do
        echo "   - $file"
    done
    echo ""
    echo "💡 Please run training first:"
    echo "   python main.py train"
    echo "   python main.py intent-train"
    exit 1
fi

echo "✅ All model files found!"

# Start Streamlit in background
echo "🚀 Starting Streamlit app..."
python -m streamlit run streamlit_demo.py --server.port 8501 --server.headless true &
STREAMLIT_PID=$!

# Wait for Streamlit to start
echo "⏳ Waiting for Streamlit to start..."
sleep 5

# Check if Streamlit is running
if ! kill -0 $STREAMLIT_PID 2>/dev/null; then
    echo "❌ Failed to start Streamlit!"
    exit 1
fi

echo "✅ Streamlit started on port 8501"

# Start ngrok tunnel
echo "🌐 Starting ngrok tunnel..."
ngrok http 8501 &
NGROK_PID=$!

# Wait for ngrok to start
echo "⏳ Waiting for ngrok to start..."
sleep 3

# Get tunnel URL
echo "🔍 Getting tunnel URL..."
sleep 2

# Try to get tunnel URL from ngrok API
TUNNEL_URL=""
if command -v curl &> /dev/null; then
    TUNNEL_URL=$(curl -s http://localhost:4040/api/tunnels | grep -o '"public_url":"[^"]*' | grep -o 'https://[^"]*' | head -1)
fi

if [ -n "$TUNNEL_URL" ]; then
    echo ""
    echo "🎉 Chatbot is now accessible via internet!"
    echo "🌐 Public URL: $TUNNEL_URL"
    echo "🔗 Local URL: http://localhost:8501"
    echo "📊 Ngrok Dashboard: http://localhost:4040"
    echo ""
    echo "💡 Press Ctrl+C to stop both Streamlit and ngrok"
    echo "💡 Share the Public URL with others to access your chatbot!"
else
    echo "⚠️ Could not get tunnel URL automatically"
    echo "💡 Check ngrok dashboard at: http://localhost:4040"
    echo "💡 Or run: ngrok http 8501"
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "👋 Shutting down..."
    echo "🛑 Stopping Streamlit (PID: $STREAMLIT_PID)..."
    kill $STREAMLIT_PID 2>/dev/null
    echo "🛑 Stopping ngrok (PID: $NGROK_PID)..."
    kill $NGROK_PID 2>/dev/null
    echo "✅ All processes stopped"
    exit 0
}

# Set trap to cleanup on exit
trap cleanup SIGINT SIGTERM

# Keep script running
echo "🔄 Keeping tunnel active..."
while true; do
    sleep 1
done
