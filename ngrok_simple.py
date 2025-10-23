"""
Script sederhana untuk menjalankan Streamlit dengan ngrok
"""

import os
import sys
import subprocess
import time
import signal
import threading

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements...")
    
    # Check virtual environment
    if "VIRTUAL_ENV" not in os.environ:
        print("❌ Virtual environment not active!")
        print("💡 Please activate 'chatbot_env' first:")
        print("   source chatbot_env/bin/activate")
        return False
    
    # Check ngrok
    try:
        subprocess.run(["ngrok", "version"], check=True, capture_output=True)
        print("✅ Ngrok found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Ngrok not found!")
        print("💡 Please install ngrok:")
        print("   - Download from: https://ngrok.com/download")
        print("   - Or install via brew: brew install ngrok")
        return False
    
    # Check model files
    model_files = [
        "models/attention_rnn_model.h5",
        "models/attention_rnn_tokenizer.pkl",
        "models/intent_classifier.pkl"
    ]
    
    missing_files = []
    for file_path in model_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing model files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\n💡 Please run training first:")
        print("   python main.py train")
        print("   python main.py intent-train")
        return False
    
    print("✅ All requirements met!")
    return True

def start_streamlit():
    """Start Streamlit app"""
    print("🚀 Starting Streamlit app...")
    try:
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "streamlit_demo.py",
            "--server.port", "8501",
            "--server.headless", "true"
        ])
        
        # Wait for Streamlit to start
        time.sleep(5)
        
        if process.poll() is None:
            print("✅ Streamlit started on port 8501")
            return process
        else:
            print("❌ Failed to start Streamlit")
            return None
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")
        return None

def start_ngrok():
    """Start ngrok tunnel"""
    print("🌐 Starting ngrok tunnel...")
    try:
        process = subprocess.Popen([
            "ngrok", "http", "8501"
        ])
        
        # Wait for ngrok to start
        time.sleep(3)
        
        if process.poll() is None:
            print("✅ Ngrok tunnel started")
            return process
        else:
            print("❌ Failed to start ngrok")
            return None
    except Exception as e:
        print(f"❌ Error starting ngrok: {e}")
        return None

def get_tunnel_url():
    """Get tunnel URL from ngrok API"""
    try:
        import requests
        response = requests.get("http://localhost:4040/api/tunnels", timeout=5)
        if response.status_code == 200:
            data = response.json()
            tunnels = data.get('tunnels', [])
            for tunnel in tunnels:
                if tunnel.get('proto') == 'https':
                    return tunnel.get('public_url')
        return None
    except Exception:
        return None

def cleanup(streamlit_process, ngrok_process):
    """Cleanup processes"""
    print("\n👋 Shutting down...")
    
    if streamlit_process:
        print("🛑 Stopping Streamlit...")
        streamlit_process.terminate()
        streamlit_process.wait()
    
    if ngrok_process:
        print("🛑 Stopping ngrok...")
        ngrok_process.terminate()
        ngrok_process.wait()
    
    print("✅ All processes stopped")

def main():
    """Main function"""
    print("🤖 Chatbot RNN Bahasa Indonesia - Ngrok Tunnel")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Start Streamlit
    streamlit_process = start_streamlit()
    if not streamlit_process:
        sys.exit(1)
    
    # Start ngrok
    ngrok_process = start_ngrok()
    if not ngrok_process:
        cleanup(streamlit_process, None)
        sys.exit(1)
    
    # Get tunnel URL
    print("🔍 Getting tunnel URL...")
    time.sleep(2)
    
    tunnel_url = get_tunnel_url()
    if tunnel_url:
        print(f"\n🎉 Chatbot is now accessible via internet!")
        print(f"🌐 Public URL: {tunnel_url}")
        print(f"🔗 Local URL: http://localhost:8501")
        print(f"📊 Ngrok Dashboard: http://localhost:4040")
        print("\n💡 Share the Public URL with others!")
    else:
        print("\n⚠️ Could not get tunnel URL automatically")
        print("💡 Check ngrok dashboard at: http://localhost:4040")
    
    print("\n💡 Press Ctrl+C to stop")
    
    # Set up signal handler
    def signal_handler(sig, frame):
        cleanup(streamlit_process, ngrok_process)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup(streamlit_process, ngrok_process)

if __name__ == "__main__":
    main()
