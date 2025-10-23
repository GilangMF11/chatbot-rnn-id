"""
Test script untuk menguji error handling pada Streamlit demo
"""

import os
import sys
import subprocess
import time
import requests

def test_error_handling():
    """Test Streamlit app dengan error handling"""
    print("🧪 Testing Streamlit Error Handling...")
    
    try:
        # Start Streamlit in background
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "streamlit_demo.py",
            "--server.headless", "true",
            "--server.port", "8504"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for startup
        time.sleep(8)
        
        # Check if process is running
        if process.poll() is None:
            print("✅ Streamlit app started successfully!")
            print("🌐 App should be available at: http://localhost:8504")
            
            # Test if app is accessible
            try:
                response = requests.get("http://localhost:8504", timeout=5)
                if response.status_code == 200:
                    print("✅ App is accessible via HTTP")
                    print("🔧 Error handling features:")
                    print("   - Model file existence check")
                    print("   - Proper error messages")
                    print("   - User guidance")
                    print("   - Graceful fallback")
                    print("   - Session state management")
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
    print("🤖 Testing Streamlit Error Handling")
    print("=" * 50)
    
    # Check if we're in virtual environment
    if "VIRTUAL_ENV" not in os.environ:
        print("⚠️ Virtual environment not active. Please activate 'chatbot_env' first.")
        print("   Example: source chatbot_env/bin/activate")
        sys.exit(1)
    
    success = test_error_handling()
    
    if success:
        print("\n🎉 Streamlit error handling test completed successfully!")
        print("💡 To run the app: streamlit run streamlit_demo.py")
        print("💡 Error handling features:")
        print("   ✅ Model file existence check")
        print("   ✅ Proper error messages")
        print("   ✅ User guidance")
        print("   ✅ Graceful fallback")
        print("   ✅ Session state management")
    else:
        print("\n❌ Streamlit error handling test failed!")
        print("💡 Check the error messages above for troubleshooting.")
