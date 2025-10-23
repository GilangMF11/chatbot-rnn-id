"""
Hybrid Chatbot: RNN + Intent Classification
Menggabungkan kelebihan RNN dan Intent Classification
"""

import os
import sys
import numpy as np
from intent_classifier import IntentClassifier
from rnn_model import IndonesianChatbotRNN
import tensorflow as tf

class HybridChatbot:
    """
    Hybrid chatbot yang menggabungkan RNN dan Intent Classification
    """
    
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.rnn_model = None
        self.tokenizer = None
        self.is_loaded = False
        
    def load_models(self):
        """
        Load both intent classifier and RNN model
        """
        # Load intent classifier
        if not self.intent_classifier.load_model("models/intent_classifier.pkl"):
            print("❌ Intent classifier not found!")
            return False
            
        # Load RNN model
        try:
            self.rnn_model = tf.keras.models.load_model("models/attention_rnn_model.h5")
            
            # Load tokenizer
            import pickle
            with open("models/attention_rnn_tokenizer.pkl", 'rb') as f:
                self.tokenizer = pickle.load(f)
                
            print("✅ Hybrid chatbot loaded successfully!")
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ RNN model loading failed: {e}")
            return False
    
    def get_response(self, user_input: str) -> str:
        """
        Get response menggunakan hybrid approach
        """
        if not self.is_loaded:
            return "Maaf, model belum dimuat."
        
        # Clean input
        user_input = user_input.strip().lower()
        
        if not user_input:
            return "Silakan ketik sesuatu untuk memulai percakapan."
        
        # 1. Coba Intent Classification dulu
        intent = self.intent_classifier.predict_intent(user_input)
        
        if intent != "unknown":
            # Jika intent ditemukan, gunakan response dari corpus
            response = self.intent_classifier.get_response(user_input)
            return f"[Intent: {intent}] {response}"
        
        # 2. Jika intent tidak ditemukan, gunakan RNN
        try:
            # Generate response dengan RNN
            rnn_response = self._generate_rnn_response(user_input)
            return f"[RNN] {rnn_response}"
            
        except Exception as e:
            return f"Maaf, terjadi kesalahan: {str(e)}"
    
    def _generate_rnn_response(self, user_input: str) -> str:
        """
        Generate response menggunakan RNN
        """
        if self.rnn_model is None or self.tokenizer is None:
            return "Model RNN tidak tersedia."
        
        # Convert input to sequence
        sequence = self.tokenizer.texts_to_sequences([user_input])
        sequence = tf.keras.preprocessing.sequence.pad_sequences(
            sequence, maxlen=30, padding='post'
        )
        
        # Generate response
        generated_text = user_input
        current_sequence = sequence[0]
        
        for _ in range(10):  # Limit response length
            # Predict next word
            prediction = self.rnn_model.predict(current_sequence.reshape(1, -1), verbose=0)
            
            # Apply temperature
            prediction = prediction / 0.7
            prediction = tf.nn.softmax(prediction)
            
            # Normalize probabilities
            probs = prediction[0].numpy()
            probs = probs / np.sum(probs)
            
            # Top-k sampling
            top_k = 5
            top_indices = np.argsort(probs)[-top_k:]
            top_probs = probs[top_indices]
            top_probs = top_probs / np.sum(top_probs)
            
            next_word_idx = np.random.choice(top_indices, p=top_probs)
            
            # Check if it's end token
            if next_word_idx == 0 or next_word_idx >= len(self.tokenizer.word_index):
                break
            
            # Add word to generated text
            if next_word_idx in self.tokenizer.index_word:
                next_word = self.tokenizer.index_word[next_word_idx]
                if next_word not in ['<OOV>', '<PAD>', '<UNK>']:
                    generated_text += " " + next_word
                else:
                    break
                    
                # Update sequence
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[-1] = next_word_idx
            else:
                break
        
        return generated_text.strip()

def demo_hybrid_chatbot():
    """
    Demo hybrid chatbot
    """
    print("🤖 Demo Hybrid Chatbot (RNN + Intent Classification)")
    print("=" * 70)
    
    chatbot = HybridChatbot()
    
    if not chatbot.load_models():
        print("❌ Failed to load models!")
        return
    
    print("\n💬 Demo Conversations:")
    print("-" * 70)
    
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
    
    print("\n" + "=" * 70)
    print("🎉 Demo selesai!")
    print("💡 Hybrid approach: Intent Classification + RNN fallback")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "chat":
        chatbot = HybridChatbot()
        if chatbot.load_models():
            print("🤖 Hybrid Chatbot (RNN + Intent Classification)")
            print("=" * 60)
            print("Ketik 'quit' untuk keluar")
            print("-" * 60)
            
            while True:
                try:
                    user_input = input("\n👤 Anda: ").strip()
                    
                    if user_input.lower() in ['quit', 'exit', 'keluar']:
                        print("👋 Sampai jumpa!")
                        break
                    
                    if user_input:
                        response = chatbot.get_response(user_input)
                        print(f"🤖 Bot: {response}")
                        
                except KeyboardInterrupt:
                    print("\n👋 Sampai jumpa!")
                    break
                except EOFError:
                    print("\n👋 Sampai jumpa!")
                    break
    else:
        demo_hybrid_chatbot()
