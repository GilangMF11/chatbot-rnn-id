#!/usr/bin/env python3
"""
Demo script untuk chatbot RNN Bahasa Indonesia
Menjalankan demo tanpa input interaktif
"""

import chatbot_interface
import sys
import os

def demo_chat():
    """Demo chatbot dengan input yang sudah ditentukan"""
    
    print("🤖 Demo Chatbot RNN Bahasa Indonesia")
    print("=" * 50)
    
    # Cek apakah model ada
    model_path = "models/attention_rnn_model.h5"
    tokenizer_path = "models/attention_rnn_tokenizer.pkl"
    
    if not os.path.exists(model_path):
        print(f"❌ Model tidak ditemukan: {model_path}")
        print("   Jalankan training terlebih dahulu dengan: python main.py train")
        return False
    
    if not os.path.exists(tokenizer_path):
        print(f"❌ Tokenizer tidak ditemukan: {tokenizer_path}")
        print("   Jalankan training terlebih dahulu dengan: python main.py train")
        return False
    
    try:
        # Load chatbot
        print("🔄 Loading chatbot...")
        chatbot = chatbot_interface.ChatbotInterface(model_path, tokenizer_path)
        print("✅ Chatbot loaded successfully!")
        
        # Demo conversations
        demo_inputs = [
            "halo",
            "apa kabar",
            "terima kasih", 
            "selamat pagi",
            "bagaimana kabarmu",
            "siapa namamu",
            "kapan kita bertemu",
            "dimana kamu tinggal",
            "mengapa kamu ada",
            "selamat malam"
        ]
        
        print("\n💬 Demo Conversations:")
        print("-" * 50)
        
        for i, user_input in enumerate(demo_inputs, 1):
            try:
                print(f"\n{i}. 👤 User: {user_input}")
                response = chatbot.generate_response(user_input)
                print(f"   🤖 Bot: {response}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        print("\n" + "=" * 50)
        print("🎉 Demo selesai!")
        print("💡 Untuk chat interaktif, gunakan: python main.py streamlit")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading chatbot: {e}")
        return False

if __name__ == "__main__":
    success = demo_chat()
    sys.exit(0 if success else 1)
