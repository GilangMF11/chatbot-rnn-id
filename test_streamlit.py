"""
Test script untuk Streamlit app
"""

import os
import sys
import subprocess
import time
import signal

def test_streamlit():
    """Test Streamlit app"""
    print("🧪 Testing Streamlit App...")
    
    try:
        # Start Streamlit in background
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.headless", "true",
            "--server.port", "8501"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for startup
        time.sleep(5)
        
        # Check if process is running
        if process.poll() is None:
            print("✅ Streamlit app started successfully!")
            print("🌐 App should be available at: http://localhost:8501")
            
            # Terminate process
            process.terminate()
            process.wait()
            print("✅ Streamlit app terminated successfully!")
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Streamlit app failed to start")
            print(f"STDOUT: {stdout.decode()}")
            print(f"STDERR: {stderr.decode()}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Streamlit: {e}")
        return False

if __name__ == "__main__":
    print("🤖 Testing Streamlit App for Chatbot Bahasa Indonesia")
    print("=" * 60)
    
    # Check if we're in virtual environment
    if "VIRTUAL_ENV" not in os.environ:
        print("⚠️ Virtual environment not active. Please activate 'chatbot_env' first.")
        print("   Example: source chatbot_env/bin/activate")
        sys.exit(1)
    
    success = test_streamlit()
    
    if success:
        print("\n🎉 Streamlit app test completed successfully!")
        print("💡 To run the app: streamlit run streamlit_app.py")
    else:
        print("\n❌ Streamlit app test failed!")
        print("💡 Check the error messages above for troubleshooting.")
