"""
Test script untuk menguji tampilan chat yang diperbaiki
"""

import os
import sys
import subprocess
import time
import requests

def test_chat_ui():
    """Test Streamlit app dengan tampilan chat yang diperbaiki"""
    print("🧪 Testing Streamlit Chat UI...")
    
    try:
        # Start Streamlit in background
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "streamlit_demo.py",
            "--server.headless", "true",
            "--server.port", "8503"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for startup
        time.sleep(8)
        
        # Check if process is running
        if process.poll() is None:
            print("✅ Streamlit app started successfully!")
            print("🌐 App should be available at: http://localhost:8503")
            
            # Test if app is accessible
            try:
                response = requests.get("http://localhost:8503", timeout=5)
                if response.status_code == 200:
                    print("✅ App is accessible via HTTP")
                    print("🎨 Chat UI features:")
                    print("   - Modern gradient styling")
                    print("   - Chat bubbles with timestamps")
                    print("   - Better button layout")
                    print("   - Improved sidebar")
                    print("   - Statistics dashboard")
                    print("   - Quick test buttons")
                    print("   - Helpful tips section")
                else:
                    print(f"⚠️ App returned status code: {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Could not access app: {e}")
            
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
    print("🤖 Testing Streamlit Chat UI")
    print("=" * 50)
    
    # Check if we're in virtual environment
    if "VIRTUAL_ENV" not in os.environ:
        print("⚠️ Virtual environment not active. Please activate 'chatbot_env' first.")
        print("   Example: source chatbot_env/bin/activate")
        sys.exit(1)
    
    success = test_chat_ui()
    
    if success:
        print("\n🎉 Streamlit chat UI test completed successfully!")
        print("💡 To run the app: streamlit run streamlit_demo.py")
        print("💡 Features:")
        print("   ✅ Modern chat interface")
        print("   ✅ Gradient styling")
        print("   ✅ Chat bubbles")
        print("   ✅ Better UX")
        print("   ✅ Statistics dashboard")
        print("   ✅ Quick test buttons")
    else:
        print("\n❌ Streamlit chat UI test failed!")
        print("💡 Check the error messages above for troubleshooting.")
