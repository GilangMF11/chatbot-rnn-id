#!/bin/bash

# Run script untuk Chatbot RNN Bahasa Indonesia

echo "🤖 Chatbot RNN Bahasa Indonesia"
echo "================================"

# Cek apakah virtual environment ada
if [ ! -d "chatbot_env" ]; then
    echo "❌ Virtual environment tidak ditemukan!"
    echo "   Jalankan: ./setup.sh terlebih dahulu"
    exit 1
fi

# Aktifkan virtual environment
source chatbot_env/bin/activate

# Tampilkan menu
echo ""
echo "Pilih opsi:"
echo "1. Check dependencies"
echo "2. Run preprocessing"
echo "3. Prepare data"
echo "4. Train model"
echo "5. Chat (console)"
echo "6. Chat (streamlit)"
echo "7. Run full pipeline"
echo "8. Demo chatbot"
echo "9. Train intent classifier"
echo "10. Demo hybrid chatbot"
echo "11. Run with ngrok tunnel"
echo "12. Exit"
echo ""

read -p "Masukkan pilihan (1-12): " choice

case $choice in
    1)
        echo "🔍 Checking dependencies..."
        python main.py check
        ;;
    2)
        echo "🔄 Running preprocessing..."
        python main.py preprocess
        ;;
    3)
        echo "📊 Preparing data..."
        python main.py prepare
        ;;
    4)
        echo "🧠 Training model..."
        python main.py train
        ;;
    5)
        echo "💬 Starting console chat..."
        python main.py chat
        ;;
    6)
        echo "🌐 Starting streamlit interface..."
        python main.py streamlit
        ;;
    7)
        echo "🚀 Running full pipeline..."
        python main.py full
        ;;
    8)
        echo "🤖 Running chatbot demo..."
        python demo_chat.py
        ;;
    9)
        echo "🧠 Training intent classifier..."
        python main.py intent-train
        ;;
    10)
        echo "🤖 Running hybrid chatbot demo..."
        python hybrid_chatbot.py
        ;;
    11)
        echo "🌐 Running with ngrok tunnel..."
        python ngrok_simple.py
        ;;
    12)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Pilihan tidak valid!"
        echo "   Pilihan yang tersedia: 1-12"
        exit 1
        ;;
esac
