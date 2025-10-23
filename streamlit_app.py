"""
Streamlit App untuk Chatbot Bahasa Indonesia
Mendukung RNN, Intent Classification, dan Hybrid approach
"""

import streamlit as st
import os
import sys
import pickle
import numpy as np
import tensorflow as tf
from datetime import datetime
import json

# Import custom modules
try:
    from intent_classifier import IntentClassifier
    from hybrid_chatbot import HybridChatbot
    from rnn_model import IndonesianChatbotRNN
    from preprocessing import IndonesianTextPreprocessor
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="🤖 Chatbot Bahasa Indonesia",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        max-width: 80%;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: auto;
        text-align: right;
    }
    .bot-message {
        background-color: #f5f5f5;
        margin-right: auto;
    }
    .model-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'chatbot_type' not in st.session_state:
    st.session_state.chatbot_type = 'hybrid'
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

class StreamlitChatbot:
    def __init__(self):
        self.hybrid_chatbot = None
        self.intent_classifier = None
        self.rnn_chatbot = None
        self.models_loaded = False
        
    def load_models(self):
        """Load all available models"""
        try:
            # Load Hybrid Chatbot
            self.hybrid_chatbot = HybridChatbot()
            if self.hybrid_chatbot.load_models():
                st.session_state.models_loaded = True
                return True
            else:
                st.error("❌ Failed to load hybrid chatbot models")
                return False
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            return False
    
    def get_response(self, user_input: str, chatbot_type: str = 'hybrid') -> str:
        """Get response from selected chatbot type"""
        if not st.session_state.models_loaded:
            return "❌ Models not loaded. Please check model files."
        
        try:
            if chatbot_type == 'hybrid' and self.hybrid_chatbot:
                return self.hybrid_chatbot.get_response(user_input)
            elif chatbot_type == 'intent' and self.hybrid_chatbot:
                # Use only intent classification
                intent = self.hybrid_chatbot.intent_classifier.predict_intent(user_input)
                if intent != "unknown":
                    return self.hybrid_chatbot.intent_classifier.get_response(user_input)
                else:
                    return "Maaf, saya tidak mengerti. Bisa dijelaskan lebih detail?"
            elif chatbot_type == 'rnn' and self.hybrid_chatbot:
                # Use only RNN
                return self.hybrid_chatbot._generate_rnn_response(user_input)
            else:
                return "❌ Chatbot type not available"
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"

# Initialize chatbot
@st.cache_resource
def initialize_chatbot():
    return StreamlitChatbot()

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 Chatbot Bahasa Indonesia</h1>', unsafe_allow_html=True)
    
    # Initialize chatbot
    chatbot = initialize_chatbot()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        st.subheader("🤖 Chatbot Type")
        chatbot_type = st.selectbox(
            "Pilih jenis chatbot:",
            ["hybrid", "intent", "rnn"],
            index=0,
            help="Hybrid: Intent + RNN, Intent: Hanya intent classification, RNN: Hanya RNN"
        )
        
        # Model status
        st.subheader("📊 Model Status")
        if st.session_state.models_loaded:
            st.markdown('<p class="status-success">✅ Models Loaded</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-error">❌ Models Not Loaded</p>', unsafe_allow_html=True)
        
        # Load models button
        if st.button("🔄 Load Models", type="primary"):
            with st.spinner("Loading models..."):
                if chatbot.load_models():
                    st.success("✅ Models loaded successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to load models!")
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation"):
            st.session_state.conversation_history = []
            st.rerun()
        
        # Model info
        st.subheader("ℹ️ Model Information")
        st.markdown("""
        **Hybrid Chatbot:**
        - Intent Classification untuk dialog terstruktur
        - RNN fallback untuk general conversation
        
        **Intent Classification:**
        - Berdasarkan corpus dialog
        - Accuracy: ~57%
        
        **RNN Model:**
        - Attention RNN
        - Vocabulary: 5,261 words
        - Training pairs: 16,782
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Chat input
        user_input = st.text_input(
            "Ketik pesan Anda:",
            placeholder="Contoh: halo, apa kabar, siapa kamu...",
            key="chat_input"
        )
        
        # Send button
        col_send, col_clear = st.columns([1, 1])
        with col_send:
            send_button = st.button("📤 Send", type="primary", use_container_width=True)
        with col_clear:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_button:
            st.session_state.conversation_history = []
            st.rerun()
        
        # Process input
        if send_button and user_input.strip():
            # Add user message to history
            st.session_state.conversation_history.append({
                "type": "user",
                "message": user_input,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            
            # Get bot response
            with st.spinner("Thinking..."):
                bot_response = chatbot.get_response(user_input, chatbot_type)
            
            # Add bot response to history
            st.session_state.conversation_history.append({
                "type": "bot",
                "message": bot_response,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            
            # Rerun to refresh the page
            st.rerun()
        
        # Display conversation
        st.subheader("📝 Conversation History")
        if st.session_state.conversation_history:
            for msg in st.session_state.conversation_history:
                if msg["type"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>👤 You ({msg['timestamp']}):</strong><br>
                        {msg['message']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message bot-message">
                        <strong>🤖 Bot ({msg['timestamp']}):</strong><br>
                        {msg['message']}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("💡 Mulai percakapan dengan mengetik pesan di atas!")
    
    with col2:
        st.header("📊 Statistics")
        
        # Conversation stats
        if st.session_state.conversation_history:
            user_messages = len([msg for msg in st.session_state.conversation_history if msg["type"] == "user"])
            bot_messages = len([msg for msg in st.session_state.conversation_history if msg["type"] == "bot"])
            
            st.metric("👤 User Messages", user_messages)
            st.metric("🤖 Bot Responses", bot_messages)
        else:
            st.metric("👤 User Messages", 0)
            st.metric("🤖 Bot Responses", 0)
        
        # Model performance
        st.subheader("🎯 Model Performance")
        if chatbot_type == "hybrid":
            st.success("✅ Best Performance")
            st.info("Intent Classification + RNN Fallback")
        elif chatbot_type == "intent":
            st.warning("⚠️ Limited to Corpus")
            st.info("Only structured dialog responses")
        else:  # rnn
            st.warning("⚠️ Experimental")
            st.info("RNN text generation")
        
        # Quick test
        st.subheader("🧪 Quick Test")
        test_inputs = ["halo", "apa kabar", "siapa kamu", "terima kasih"]
        for test_input in test_inputs:
            if st.button(f"Test: {test_input}", key=f"test_{test_input}"):
                with st.spinner("Testing..."):
                    response = chatbot.get_response(test_input, chatbot_type)
                st.text_area(f"Response to '{test_input}':", response, height=100, key=f"response_{test_input}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🤖 Chatbot Bahasa Indonesia - Hybrid Approach (Intent Classification + RNN)</p>
    <p>Built with Streamlit, TensorFlow, and Scikit-learn</p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
