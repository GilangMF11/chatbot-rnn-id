"""
Script untuk menjalankan Streamlit dengan ngrok tunneling
"""

import os
import sys
import subprocess
import time
import threading
import signal
import requests
from urllib.parse import urlparse

class NgrokTunnel:
    def __init__(self, streamlit_port=8501, ngrok_port=4040):
        self.streamlit_port = streamlit_port
        self.ngrok_port = ngrok_port
        self.streamlit_process = None
        self.ngrok_process = None
        self.tunnel_url = None
        
    def start_streamlit(self):
        """Start Streamlit app"""
        print("🚀 Starting Streamlit app...")
        try:
            self.streamlit_process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", "streamlit_demo.py",
                "--server.port", str(self.streamlit_port),
                "--server.headless", "true"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for Streamlit to start
            time.sleep(5)
            
            if self.streamlit_process.poll() is None:
                print(f"✅ Streamlit started on port {self.streamlit_port}")
                return True
            else:
                print("❌ Failed to start Streamlit")
                return False
        except Exception as e:
            print(f"❌ Error starting Streamlit: {e}")
            return False
    
    def start_ngrok(self):
        """Start ngrok tunnel"""
        print("🌐 Starting ngrok tunnel...")
        try:
            self.ngrok_process = subprocess.Popen([
                "ngrok", "http", str(self.streamlit_port)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for ngrok to start
            time.sleep(3)
            
            # Get tunnel URL from ngrok API
            try:
                response = requests.get(f"http://localhost:{self.ngrok_port}/api/tunnels")
                if response.status_code == 200:
                    data = response.json()
                    tunnels = data.get('tunnels', [])
                    for tunnel in tunnels:
                        if tunnel.get('proto') == 'https':
                            self.tunnel_url = tunnel.get('public_url')
                            break
                    
                    if self.tunnel_url:
                        print(f"✅ Ngrok tunnel created: {self.tunnel_url}")
                        return True
                    else:
                        print("❌ Failed to get ngrok tunnel URL")
                        return False
                else:
                    print("❌ Failed to connect to ngrok API")
                    return False
            except Exception as e:
                print(f"❌ Error getting tunnel URL: {e}")
                return False
        except Exception as e:
            print(f"❌ Error starting ngrok: {e}")
            return False
    
    def stop_processes(self):
        """Stop all processes"""
        print("🛑 Stopping processes...")
        
        if self.streamlit_process:
            self.streamlit_process.terminate()
            self.streamlit_process.wait()
            print("✅ Streamlit stopped")
        
        if self.ngrok_process:
            self.ngrok_process.terminate()
            self.ngrok_process.wait()
            print("✅ Ngrok stopped")
    
    def run(self):
        """Run Streamlit with ngrok tunnel"""
        print("🤖 Starting Chatbot with Ngrok Tunnel")
        print("=" * 50)
        
        # Check if ngrok is installed
        try:
            subprocess.run(["ngrok", "version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Ngrok not found!")
            print("💡 Please install ngrok first:")
            print("   - Download from: https://ngrok.com/download")
            print("   - Or install via brew: brew install ngrok")
            return False
        
        # Start Streamlit
        if not self.start_streamlit():
            return False
        
        # Start ngrok
        if not self.start_ngrok():
            self.stop_processes()
            return False
        
        print("\n🎉 Chatbot is now accessible via internet!")
        print(f"🌐 Public URL: {self.tunnel_url}")
        print(f"🔗 Local URL: http://localhost:{self.streamlit_port}")
        print(f"📊 Ngrok Dashboard: http://localhost:{self.ngrok_port}")
        print("\n💡 Press Ctrl+C to stop")
        
        try:
            # Keep running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Shutting down...")
            self.stop_processes()
            return True

def main():
    """Main function"""
    # Check if we're in virtual environment
    if "VIRTUAL_ENV" not in os.environ:
        print("⚠️ Virtual environment not active. Please activate 'chatbot_env' first.")
        print("   Example: source chatbot_env/bin/activate")
        sys.exit(1)
    
    # Check if model files exist
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
        sys.exit(1)
    
    # Start tunnel
    tunnel = NgrokTunnel()
    tunnel.run()

if __name__ == "__main__":
    main()
