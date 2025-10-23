"""
Interface chatbot untuk berinteraksi dengan model RNN
Mendukung berbagai mode interaksi dan visualisasi
"""

import numpy as np
import tensorflow as tf
import pickle
import json
from typing import List, Dict, Optional
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from rnn_model import IndonesianChatbotRNN
from preprocessing import IndonesianTextPreprocessor

class ChatbotInterface:
    def __init__(self, model_path: str, tokenizer_path: str):
        """
        Inisialisasi interface chatbot
        
        Args:
            model_path: Path ke model yang sudah dilatih
            tokenizer_path: Path ke tokenizer
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.chatbot = None
        self.preprocessor = None
        self.conversation_history = []
        
        # Load model dan tokenizer
        self.load_model()
        self.load_preprocessor()
    
    def load_model(self) -> None:
        """
        Load model dan tokenizer
        """
        try:
            # Load tokenizer
            with open(self.tokenizer_path, 'rb') as f:
                tokenizer = pickle.load(f)
            
            # Get model info
            vocab_size = len(tokenizer.word_index) + 1
            max_sequence_length = 30  # Default, bisa disesuaikan
            
            # Create chatbot instance
            self.chatbot = IndonesianChatbotRNN(vocab_size, max_sequence_length)
            self.chatbot.load_model(self.model_path)
            self.chatbot.load_tokenizer(self.tokenizer_path)
            
            print("Model dan tokenizer berhasil dimuat!")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def load_preprocessor(self) -> None:
        """
        Load preprocessor untuk input text
        """
        try:
            self.preprocessor = IndonesianTextPreprocessor(
                'dataset/normalization/normalization.txt',
                'dataset/normalization/stopword.txt'
            )
            print("Preprocessor berhasil dimuat!")
        except Exception as e:
            print(f"Error loading preprocessor: {str(e)}")
            # Fallback preprocessor
            self.preprocessor = None
    
    def preprocess_input(self, text: str) -> str:
        """
        Preprocess input text
        """
        if self.preprocessor:
            return self.preprocessor.preprocess_text(text)
        else:
            # Simple preprocessing fallback
            return text.lower().strip()
    
    def generate_response(self, user_input: str, max_length: int = 30, 
                        temperature: float = 0.8) -> str:
        """
        Generate response untuk user input
        """
        try:
            # Preprocess input
            processed_input = self.preprocess_input(user_input)
            
            # Generate response
            response = self.chatbot.generate_text(
                processed_input, 
                max_length=max_length, 
                temperature=temperature
            )
            
            # Clean response
            response = response.replace(processed_input, "").strip()
            if not response:
                response = "Maaf, saya tidak bisa memberikan respons yang tepat."
            
            return response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def chat(self, user_input: str, max_length: int = 30, 
             temperature: float = 0.8) -> Dict:
        """
        Chat dengan chatbot dan simpan history
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Generate response
        response = self.generate_response(user_input, max_length, temperature)
        
        # Save to history
        chat_entry = {
            'timestamp': timestamp,
            'user_input': user_input,
            'bot_response': response,
            'max_length': max_length,
            'temperature': temperature
        }
        
        self.conversation_history.append(chat_entry)
        
        return chat_entry
    
    def get_conversation_history(self) -> List[Dict]:
        """
        Get conversation history
        """
        return self.conversation_history
    
    def clear_history(self) -> None:
        """
        Clear conversation history
        """
        self.conversation_history = []
    
    def save_conversation(self, filepath: str) -> None:
        """
        Save conversation history ke file
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
    
    def load_conversation(self, filepath: str) -> None:
        """
        Load conversation history dari file
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            self.conversation_history = json.load(f)

def create_streamlit_app():
    """
    Buat Streamlit app untuk chatbot interface
    """
    st.set_page_config(
        page_title="Chatbot RNN Bahasa Indonesia",
        page_icon="🤖",
        layout="wide"
    )
    
    # Header
    st.title("🤖 Chatbot RNN Bahasa Indonesia")
    st.markdown("Chatbot yang dilatih menggunakan Recurrent Neural Network (RNN) untuk bahasa Indonesia")
    
    # Sidebar untuk konfigurasi
    st.sidebar.header("⚙️ Konfigurasi Model")
    
    # Model selection
    model_options = {
        "Simple LSTM": ("models/simple_lstm_model.h5", "models/simple_lstm_tokenizer.pkl"),
        "Bidirectional LSTM": ("models/bidirectional_lstm_model.h5", "models/bidirectional_lstm_tokenizer.pkl"),
        "Attention LSTM": ("models/attention_lstm_model.h5", "models/attention_lstm_tokenizer.pkl")
    }
    
    selected_model = st.sidebar.selectbox(
        "Pilih Model:",
        list(model_options.keys())
    )
    
    model_path, tokenizer_path = model_options[selected_model]
    
    # Check if model exists
    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        st.error(f"Model {selected_model} belum tersedia. Silakan jalankan training terlebih dahulu.")
        st.stop()
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        try:
            st.session_state.chatbot = ChatbotInterface(model_path, tokenizer_path)
            st.session_state.conversation_history = []
        except Exception as e:
            st.error(f"Error loading chatbot: {str(e)}")
            st.stop()
    
    chatbot = st.session_state.chatbot
    
    # Parameters
    st.sidebar.subheader("🎛️ Parameter Generation")
    
    max_length = st.sidebar.slider(
        "Max Length:",
        min_value=10,
        max_value=100,
        value=30,
        help="Maksimal panjang respons"
    )
    
    temperature = st.sidebar.slider(
        "Temperature:",
        min_value=0.1,
        max_value=2.0,
        value=0.8,
        step=0.1,
        help="Kontrol kreativitas respons (lebih tinggi = lebih kreatif)"
    )
    
    # Main chat interface
    st.subheader("💬 Chat dengan Bot")
    
    # Chat input
    user_input = st.text_input(
        "Ketik pesan Anda:",
        placeholder="Contoh: Halo, apa kabar?",
        key="user_input"
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        send_button = st.button("Kirim", type="primary")
    
    with col2:
        if st.button("Clear History"):
            chatbot.clear_history()
            st.session_state.conversation_history = []
            st.rerun()
    
    # Process input
    if send_button and user_input:
        with st.spinner("Bot sedang berpikir..."):
            chat_entry = chatbot.chat(user_input, max_length, temperature)
            st.session_state.conversation_history.append(chat_entry)
    
    # Display conversation
    st.subheader("📝 Riwayat Percakapan")
    
    if st.session_state.conversation_history:
        for i, entry in enumerate(reversed(st.session_state.conversation_history[-10:])):  # Show last 10
            with st.container():
                # User message
                st.markdown(f"**👤 Anda ({entry['timestamp']}):**")
                st.info(entry['user_input'])
                
                # Bot response
                st.markdown(f"**🤖 Bot:**")
                st.success(entry['bot_response'])
                
                st.markdown("---")
    else:
        st.info("Belum ada percakapan. Mulai chat dengan bot!")
    
    # Statistics
    st.subheader("📊 Statistik Percakapan")
    
    if st.session_state.conversation_history:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Percakapan", len(st.session_state.conversation_history))
        
        with col2:
            avg_user_length = np.mean([len(entry['user_input'].split()) for entry in st.session_state.conversation_history])
            st.metric("Rata-rata Kata User", f"{avg_user_length:.1f}")
        
        with col3:
            avg_bot_length = np.mean([len(entry['bot_response'].split()) for entry in st.session_state.conversation_history])
            st.metric("Rata-rata Kata Bot", f"{avg_bot_length:.1f}")
    
    # Export conversation
    if st.session_state.conversation_history:
        st.subheader("💾 Export Percakapan")
        
        if st.button("Download Conversation"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"
            
            chatbot.save_conversation(filename)
            
            with open(filename, 'r', encoding='utf-8') as f:
                st.download_button(
                    label="Download JSON",
                    data=f.read(),
                    file_name=filename,
                    mime="application/json"
                )
            
            # Clean up
            os.remove(filename)

def create_console_interface():
    """
    Buat console interface untuk chatbot
    """
    print("🤖 Chatbot RNN Bahasa Indonesia")
    print("=" * 50)
    print("Ketik 'quit' untuk keluar")
    print("Ketik 'clear' untuk menghapus riwayat")
    print("Ketik 'stats' untuk melihat statistik")
    print("=" * 50)
    
    # Initialize chatbot
    try:
        # Try to load the best available model
        model_files = [
            ("models/attention_lstm_model.h5", "models/attention_lstm_tokenizer.pkl"),
            ("models/bidirectional_lstm_model.h5", "models/bidirectional_lstm_tokenizer.pkl"),
            ("models/simple_lstm_model.h5", "models/simple_lstm_tokenizer.pkl")
        ]
        
        chatbot = None
        for model_path, tokenizer_path in model_files:
            if os.path.exists(model_path) and os.path.exists(tokenizer_path):
                chatbot = ChatbotInterface(model_path, tokenizer_path)
                print(f"✅ Model loaded: {model_path}")
                break
        
        if chatbot is None:
            print("❌ No trained model found. Please run training first.")
            return
            
    except Exception as e:
        print(f"❌ Error loading chatbot: {str(e)}")
        return
    
    # Chat loop
    while True:
        try:
            user_input = input("\n👤 Anda: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Sampai jumpa!")
                break
            elif user_input.lower() == 'clear':
                chatbot.clear_history()
                print("🗑️ Riwayat percakapan dihapus")
                continue
            elif user_input.lower() == 'stats':
                history = chatbot.get_conversation_history()
                print(f"📊 Total percakapan: {len(history)}")
                if history:
                    avg_length = np.mean([len(entry['bot_response'].split()) for entry in history])
                    print(f"📊 Rata-rata panjang respons: {avg_length:.1f} kata")
                continue
            elif not user_input:
                continue
            
            # Generate response
            print("🤖 Bot: ", end="", flush=True)
            response = chatbot.generate_response(user_input)
            print(response)
            
            # Save to history
            chatbot.chat(user_input)
            
        except KeyboardInterrupt:
            print("\n👋 Sampai jumpa!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def main():
    """
    Main function untuk menjalankan interface
    """
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'streamlit':
        # Run Streamlit app
        create_streamlit_app()
    else:
        # Run console interface
        create_console_interface()

if __name__ == "__main__":
    main()
