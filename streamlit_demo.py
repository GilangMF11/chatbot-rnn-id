"""
Demo Streamlit app yang lebih sederhana untuk menguji fitur send
"""

import streamlit as st
import os
import sys
from datetime import datetime

# Import custom modules
try:
    from intent_classifier import IntentClassifier
    from hybrid_chatbot import HybridChatbot
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="🤖 Chatbot Demo",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'chatbot_loaded' not in st.session_state:
    st.session_state.chatbot_loaded = False
if 'chatbot_instance' not in st.session_state:
    st.session_state.chatbot_instance = None

class SimpleChatbot:
    def __init__(self):
        self.hybrid_chatbot = None
        
    def load_models(self):
        """Load chatbot models"""
        try:
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
                st.error(f"❌ Missing model files: {', '.join(missing_files)}")
                st.info("💡 Please run training first: python main.py train")
                return False
            
            # Initialize chatbot
            self.hybrid_chatbot = HybridChatbot()
            
            # Load models
            if self.hybrid_chatbot.load_models():
                st.session_state.chatbot_loaded = True
                return True
            else:
                st.error("❌ Failed to load models")
                return False
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            return False
    
    def get_response(self, user_input: str, model_type: str = "Hybrid (Intent + RNN)") -> str:
        """Get response from chatbot"""
        if not st.session_state.chatbot_loaded or self.hybrid_chatbot is None:
            return "❌ Models not loaded. Please click 'Load Models' button first."
        
        try:
            # Convert model_type to hybrid chatbot format
            if model_type == "Hybrid (Intent + RNN)":
                return self.hybrid_chatbot.get_response(user_input)
            elif model_type == "Intent Only":
                # Use only intent classification
                intent = self.hybrid_chatbot.intent_classifier.predict_intent(user_input)
                if intent != "unknown":
                    return self.hybrid_chatbot.intent_classifier.get_response(user_input)
                else:
                    return "Maaf, saya tidak mengerti. Bisa dijelaskan lebih detail?"
            elif model_type == "RNN Only":
                # Use only RNN
                return self.hybrid_chatbot._generate_rnn_response(user_input)
            else:
                return self.hybrid_chatbot.get_response(user_input)
        except Exception as e:
            return f"❌ Error: {str(e)}"

def main():
    st.title("🤖 Chatbot Bahasa Indonesia - Demo")
    
    # Initialize chatbot from session state
    if st.session_state.chatbot_instance is None:
        st.session_state.chatbot_instance = SimpleChatbot()
    
    chatbot = st.session_state.chatbot_instance
    
    # Sidebar with better styling
    with st.sidebar:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            text-align: center;
        ">
            <h3 style="color: white; margin: 0;">⚙️ Settings</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Model selection
        st.subheader("🤖 Model Selection")
        model_type = st.selectbox(
            "Pilih jenis model:",
            ["Hybrid (Intent + RNN)", "Intent Only", "RNN Only"],
            index=0,
            help="Hybrid: Intent Classification + RNN, Intent: Hanya intent classification, RNN: Hanya RNN"
        )
        
        # Load models button
        if st.button("🔄 Load Models", type="primary", use_container_width=True):
            with st.spinner("🔄 Loading models..."):
                if chatbot.load_models():
                    st.success("✅ Models loaded successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to load models!")
        
        # Model status with better styling
        if st.session_state.chatbot_loaded:
            st.markdown("""
            <div style="
                background: #d4edda;
                color: #155724;
                padding: 0.5rem;
                border-radius: 5px;
                text-align: center;
                margin: 0.5rem 0;
            ">
                ✅ Models Loaded
            </div>
            """, unsafe_allow_html=True)
            
            # Show current model type
            st.info(f"🤖 Current Model: {model_type}")
        else:
            st.markdown("""
            <div style="
                background: #f8d7da;
                color: #721c24;
                padding: 0.5rem;
                border-radius: 5px;
                text-align: center;
                margin: 0.5rem 0;
            ">
                ⚠️ Models Not Loaded
            </div>
            """, unsafe_allow_html=True)
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.conversation_history = []
            st.rerun()
        
        # Add some spacing
        st.markdown("---")
        
        # Quick actions
        st.subheader("🚀 Quick Actions")
        if st.button("💬 Start New Chat", use_container_width=True):
            st.session_state.conversation_history = []
            st.rerun()
        
        if st.button("🔄 Refresh Page", use_container_width=True):
            st.rerun()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Chat input with better styling
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <h4 style="color: white; margin: 0 0 0.5rem 0;">💬 Kirim Pesan</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Chat input
        user_input = st.text_input(
            "Ketik pesan Anda:",
            placeholder="Contoh: halo, apa kabar, siapa kamu...",
            key="chat_input",
            help="Tekan Enter atau klik Send untuk mengirim pesan"
        )
        
        # Send and Clear buttons with better styling
        col_send, col_clear, col_space = st.columns([2, 1, 1])
        with col_send:
            send_button = st.button(
                "📤 Send Message", 
                type="primary", 
                use_container_width=True,
                help="Kirim pesan ke chatbot"
            )
        with col_clear:
            clear_button = st.button(
                "🗑️ Clear Chat", 
                use_container_width=True,
                help="Hapus semua percakapan"
            )
        
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
            
            # Get bot response with better loading animation
            with st.spinner("🤖 Bot sedang berpikir..."):
                import time
                time.sleep(0.5)  # Small delay for better UX
                bot_response = chatbot.get_response(user_input, model_type)
            
            # Add bot response to history
            st.session_state.conversation_history.append({
                "type": "bot",
                "message": bot_response,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            
            # Rerun to refresh
            st.rerun()
        
        # Display conversation with better styling
        st.subheader("💬 Chat History")
        
        # Create a container for chat messages
        chat_container = st.container()
        
        with chat_container:
            if st.session_state.conversation_history:
                # Display messages in reverse order (newest first) for better UX
                for i, msg in enumerate(reversed(st.session_state.conversation_history)):
                    if msg["type"] == "user":
                        # User message - right aligned, blue background
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            color: white;
                            padding: 12px 16px;
                            border-radius: 18px 18px 4px 18px;
                            margin: 8px 0;
                            margin-left: 20%;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                            word-wrap: break-word;
                        ">
                            <div style="font-size: 0.8em; opacity: 0.8; margin-bottom: 4px;">
                                👤 You • {msg['timestamp']}
                            </div>
                            <div style="font-size: 1em;">
                                {msg['message']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Bot message - left aligned, light background
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                            color: #333;
                            padding: 12px 16px;
                            border-radius: 18px 18px 18px 4px;
                            margin: 8px 0;
                            margin-right: 20%;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                            word-wrap: break-word;
                        ">
                            <div style="font-size: 0.8em; opacity: 0.7; margin-bottom: 4px;">
                                🤖 Bot • {msg['timestamp']}
                            </div>
                            <div style="font-size: 1em;">
                                {msg['message']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                # Welcome message
                st.markdown("""
                <div style="
                    text-align: center;
                    padding: 2rem;
                    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    color: white;
                    border-radius: 15px;
                    margin: 1rem 0;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                ">
                    <h3 style="margin: 0 0 1rem 0;">🤖 Selamat Datang!</h3>
                    <p style="margin: 0; opacity: 0.9;">
                        Mulai percakapan dengan mengetik pesan di atas.<br>
                        Saya siap membantu Anda!
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            text-align: center;
        ">
            <h3 style="color: white; margin: 0;">📊 Statistics</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Conversation stats with better styling
        if st.session_state.conversation_history:
            user_messages = len([msg for msg in st.session_state.conversation_history if msg["type"] == "user"])
            bot_messages = len([msg for msg in st.session_state.conversation_history if msg["type"] == "bot"])
            
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                st.metric("👤 User Messages", user_messages, delta=None)
            with col_metric2:
                st.metric("🤖 Bot Responses", bot_messages, delta=None)
            
            # Show last message
            if st.session_state.conversation_history:
                last_msg = st.session_state.conversation_history[-1]
                st.markdown("**💬 Last Message:**")
                if last_msg["type"] == "user":
                    st.info(f"👤 You: {last_msg['message'][:50]}...")
                else:
                    st.success(f"🤖 Bot: {last_msg['message'][:50]}...")
        else:
            st.metric("👤 User Messages", 0)
            st.metric("🤖 Bot Responses", 0)
            st.info("💡 Start chatting to see statistics!")
        
        # Quick test with better styling
        st.markdown("---")
        st.subheader("🧪 Quick Test")
        st.markdown("Test chatbot dengan input yang sudah ditentukan:")
        
        test_inputs = ["halo", "apa kabar", "siapa kamu"]
        for test_input in test_inputs:
            if st.button(f"Test: {test_input}", key=f"test_{test_input}", use_container_width=True):
                with st.spinner("🤖 Testing..."):
                    response = chatbot.get_response(test_input, model_type)
                st.text_area(f"Response to '{test_input}':", response, height=60, key=f"response_{test_input}")
        
        # Add some helpful tips
        st.markdown("---")
        st.markdown("""
        <div style="
            background: #e3f2fd;
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #2196f3;
        ">
            <h4 style="margin: 0 0 0.5rem 0;">💡 Tips</h4>
            <ul style="margin: 0; padding-left: 1rem;">
                <li>Load models terlebih dahulu</li>
                <li>Ketik pesan dan klik Send</li>
                <li>Gunakan Quick Test untuk demo</li>
                <li>Clear chat untuk memulai baru</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
