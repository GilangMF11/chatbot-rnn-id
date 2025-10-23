#!/bin/bash

# Setup script untuk Chatbot RNN Bahasa Indonesia
# Pastikan menggunakan Python 3.11

echo "🤖 Chatbot RNN Bahasa Indonesia - Setup"
echo "========================================"

# Cek Python version
echo "🔍 Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $python_version"

if [[ "$python_version" != "3.11" ]]; then
    echo "⚠️  WARNING: Python 3.11 recommended for best compatibility"
    echo "   Current version: $python_version"
    echo "   Install Python 3.11: brew install python@3.11"
fi

# Buat virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv chatbot_env
source chatbot_env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

echo "✅ Setup completed!"
echo ""
echo "🚀 Next steps:"
echo "1. Activate environment: source chatbot_env/bin/activate"
echo "2. Check dependencies: python main.py check"
echo "3. Run full pipeline: python main.py full"
echo "4. Start chat: python main.py chat"
