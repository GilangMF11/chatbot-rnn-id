"""
Intent-based Chatbot untuk Bahasa Indonesia
Menggunakan corpus dialog yang terstruktur
"""

import os
import sys
from intent_classifier import IntentClassifier

class IntentChatbot:
    """
    Chatbot berbasis intent classification
    """
    
    def __init__(self, model_path: str = "models/intent_classifier.pkl"):
        self.classifier = IntentClassifier()
        self.model_path = model_path
        self.is_loaded = False
        
    def load_model(self):
        """
        Load intent classifier model
        """
        if self.classifier.load_model(self.model_path):
            self.is_loaded = True
            print("✅ Intent chatbot loaded successfully!")
            return True
        else:
            print("❌ Failed to load intent classifier!")
            print("   Run: python intent_classifier.py to train the model")
            return False
    
    def get_response(self, user_input: str) -> str:
        """
        Get response dari user input
        """
        if not self.is_loaded:
            return "Maaf, model belum dimuat. Silakan jalankan training terlebih dahulu."
        
        # Clean input
        user_input = user_input.strip().lower()
        
        if not user_input:
            return "Silakan ketik sesuatu untuk memulai percakapan."
        
        # Get response berdasarkan intent
        response = self.classifier.get_response(user_input)
        return response
    
    def chat_console(self):
        """
        Console chat interface
        """
        if not self.load_model():
            return
        
        print("🤖 Intent-based Chatbot Bahasa Indonesia")
        print("=" * 50)
        print("Ketik 'quit' untuk keluar")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n👤 Anda: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'keluar']:
                    print("👋 Sampai jumpa!")
                    break
                
                if user_input:
                    response = self.get_response(user_input)
                    print(f"🤖 Bot: {response}")
                    
            except KeyboardInterrupt:
                print("\n👋 Sampai jumpa!")
                break
            except EOFError:
                print("\n👋 Sampai jumpa!")
                break

def demo_intent_chatbot():
    """
    Demo intent chatbot
    """
    print("🤖 Demo Intent-based Chatbot Bahasa Indonesia")
    print("=" * 60)
    
    chatbot = IntentChatbot()
    
    if not chatbot.load_model():
        print("❌ Model tidak ditemukan!")
        print("   Jalankan: python intent_classifier.py")
        return
    
    print("\n💬 Demo Conversations:")
    print("-" * 50)
    
    demo_inputs = [
        "halo",
        "apa kabar",
        "siapa kamu",
        "terima kasih",
        "selamat pagi",
        "bagaimana kabarmu",
        "kapan kita bertemu",
        "dimana kamu tinggal",
        "mengapa kamu ada",
        "selamat malam"
    ]
    
    for i, user_input in enumerate(demo_inputs, 1):
        response = chatbot.get_response(user_input)
        print(f"{i:2d}. 👤 User: {user_input}")
        print(f"    🤖 Bot: {response}")
    
    print("\n" + "=" * 60)
    print("🎉 Demo selesai!")
    print("💡 Untuk chat interaktif, gunakan: python intent_chatbot.py chat")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "chat":
        chatbot = IntentChatbot()
        chatbot.chat_console()
    else:
        demo_intent_chatbot()
