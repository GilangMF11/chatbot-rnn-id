"""
Main script untuk Chatbot RNN Bahasa Indonesia
Mengintegrasikan seluruh pipeline dari preprocessing hingga interface

Cara penggunaan:
- python main.py check          # Cek dependencies
- python main.py preprocess     # Jalankan preprocessing
- python main.py prepare        # Siapkan data
- python main.py train          # Training model
- python main.py chat           # Chat interface
- python main.py streamlit      # Streamlit interface
- python main.py full           # Jalankan semua pipeline
"""

import os
import sys
import argparse
from typing import Optional
import subprocess

def run_preprocessing():
    """Jalankan preprocessing pipeline"""
    print("🔄 Running preprocessing pipeline...")
    try:
        import preprocessing
        print("✅ Preprocessing completed!")
        return True
    except Exception as e:
        print(f"❌ Error in preprocessing: {str(e)}")
        return False

def run_data_preparation():
    """Jalankan data preparation"""
    print("🔄 Running data preparation...")
    try:
        import data_preparation
        print("✅ Data preparation completed!")
        return True
    except Exception as e:
        print(f"❌ Error in data preparation: {str(e)}")
        return False

def run_training():
    """Jalankan training pipeline"""
    print("🔄 Running training pipeline...")
    try:
        import training_pipeline
        print("✅ Training completed!")
        return True
    except Exception as e:
        print(f"❌ Error in training: {str(e)}")
        return False

def run_chatbot_interface(interface_type: str = "console"):
    """Jalankan chatbot interface"""
    print(f"🔄 Starting chatbot interface ({interface_type})...")
    try:
        if interface_type == "streamlit":
            # Run Streamlit app
            subprocess.run([sys.executable, "-m", "streamlit", "run", "chatbot_interface.py", "--", "streamlit"])
        else:
            # Run console interface
            import chatbot_interface
            chatbot_interface.create_console_interface()
        return True
    except Exception as e:
        print(f"❌ Error in chatbot interface: {str(e)}")
        return False

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'tensorflow', 'numpy', 'pandas', 'sklearn', 
        'matplotlib', 'seaborn', 'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("📦 Install missing packages with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Chatbot RNN Bahasa Indonesia")
    parser.add_argument(
        "command", 
        choices=["preprocess", "prepare", "train", "chat", "streamlit", "full", "check"],
        help="Command to run"
    )
    parser.add_argument(
        "--interface", 
        choices=["console", "streamlit"], 
        default="console",
        help="Interface type for chat command"
    )
    
    args = parser.parse_args()
    
    print("🤖 Chatbot RNN Bahasa Indonesia")
    print("=" * 50)
    
    if args.command == "check":
        success = check_dependencies()
        sys.exit(0 if success else 1)
    
    if args.command == "preprocess":
        success = run_preprocessing()
        sys.exit(0 if success else 1)
    
    elif args.command == "prepare":
        success = run_data_preparation()
        sys.exit(0 if success else 1)
    
    elif args.command == "train":
        success = run_training()
        sys.exit(0 if success else 1)
    
    elif args.command == "chat":
        success = run_chatbot_interface(args.interface)
        sys.exit(0 if success else 1)
    
    elif args.command == "streamlit":
        success = run_chatbot_interface("streamlit")
        sys.exit(0 if success else 1)
    
    elif args.command == "full":
        print("🚀 Running full pipeline...")
        
        # Check dependencies
        if not check_dependencies():
            sys.exit(1)
        
        # Run all steps
        steps = [
            ("Preprocessing", run_preprocessing),
            ("Data Preparation", run_data_preparation),
            ("Training", run_training),
        ]
        
        for step_name, step_func in steps:
            print(f"\n{'='*20} {step_name} {'='*20}")
            if not step_func():
                print(f"❌ {step_name} failed!")
                sys.exit(1)
        
        print("\n🎉 Full pipeline completed successfully!")
        print("💬 You can now run: python main.py chat")
        
        # Ask if user wants to start chat
        try:
            start_chat = input("\nStart chatbot interface? (y/n): ").lower().strip()
            if start_chat in ['y', 'yes']:
                run_chatbot_interface(args.interface)
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()