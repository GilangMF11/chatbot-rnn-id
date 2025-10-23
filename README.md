# 🤖 Chatbot RNN Bahasa Indonesia

Chatbot berbasis Recurrent Neural Network (RNN) untuk bahasa Indonesia yang menggunakan dataset lengkap termasuk corpus, sentiment, dan entities. Menggunakan **Hybrid Approach** yang menggabungkan Intent Classification dan RNN untuk performa optimal.

## 📊 Dataset yang Digunakan

### ✅ Dataset Lengkap (5,305 items)
- **Corpus**: 66 items (dialog, agent, user, motivasi)
- **Sentiment**: 5,238 items (positive: 1,549, negative: 3,689)
- **Entities**: 1 item (example)
- **Total Training Pairs**: 16,782 pairs

### 📁 Struktur Dataset
```
dataset/
├── corpus/id/           # Dialog corpus
├── sentiment/tsv/        # Sentiment data (positive/negative)
├── entities/json/        # Entity data
├── normalization/        # Text normalization
└── notNormalized/        # Slang words & emoticons
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone repository
git clone <repository-url>
cd chatbot-rnn-id

# Setup environment (Python 3.11)
chmod +x setup.sh
./setup.sh

# Activate virtual environment
source chatbot_env/bin/activate
```

### 2. Run Pipeline
```bash
# Option 1: Interactive menu
./run.sh

# Option 2: Direct commands
python main.py check          # Check dependencies
python main.py preprocess     # Run preprocessing
python main.py prepare        # Prepare data
python main.py train          # Train model
python main.py chat           # Console chat
python main.py streamlit      # Web interface
python main.py full           # Run full pipeline
```

### 3. Demo Chatbot
```bash
# Non-interactive demo
python demo_chat.py

# Interactive chat
python main.py chat
```

## 🏗️ Architecture

### Model Types
- **SimpleRNN**: Basic RNN implementation
- **Bidirectional RNN**: Bidirectional processing
- **Attention RNN**: RNN with attention mechanism

### Optimizations
- **Embedding**: 256 dimensions (2x original)
- **RNN Units**: 256 units
- **Dense Layers**: 512 → 256 → vocab_size
- **Dropout**: 0.3 (input + recurrent)
- **Learning Rate**: 0.0005 (reduced from 0.001)
- **Training**: 100 epochs, batch size 16

### Text Generation
- **Top-k Sampling**: k=10 for coherent responses
- **Temperature**: 0.7 for balanced creativity
- **Max Length**: 20 words (focused responses)
- **Token Filtering**: Removes OOV/PAD/UNK tokens

## 📈 Performance

### Training Results
- **SimpleRNN**: ~58% accuracy
- **Bidirectional RNN**: ~58% accuracy  
- **Attention RNN**: 47.83% accuracy

### Dataset Statistics
- **Vocabulary Size**: 5,261 words
- **Training Pairs**: 16,782 pairs
- **Average Input Length**: 1.76 words
- **Average Output Length**: 2.16 words

## 🔧 Features

### Hybrid Approach
- ✅ **Intent Classification**: Untuk dialog terstruktur (accuracy ~57%)
- ✅ **RNN Fallback**: Untuk general conversation
- ✅ **Smart Routing**: Otomatis pilih model terbaik
- ✅ **Best Performance**: Menggabungkan kelebihan kedua approach

### Text Preprocessing
- ✅ Normalization (slang → formal)
- ✅ Stopword removal
- ✅ Emoticon sentiment mapping
- ✅ Enhanced training pairs with context

### Model Features
- ✅ Pure RNN implementation (no LSTM)
- ✅ Multiple model architectures
- ✅ Optimized hyperparameters
- ✅ Advanced text generation

### Interface Features
- ✅ **Streamlit Web UI**: Modern web interface
- ✅ **Model Selection**: Pilih jenis chatbot (Hybrid/Intent/RNN)
- ✅ **Real-time Chat**: Interactive conversation
- ✅ **Statistics**: Performance metrics dan conversation stats
- ✅ **Quick Test**: Test dengan input yang sudah ditentukan

### Interface Options
- ✅ Console interface
- ✅ Streamlit web interface
- ✅ Non-interactive demo
- ✅ Batch processing

## 📁 Project Structure

```
chatbot-rnn-id/
├── main.py                 # Main entry point
├── preprocessing.py        # Text preprocessing
├── data_preparation.py     # Data preparation
├── rnn_model.py           # RNN model implementation
├── training_pipeline.py   # Training pipeline
├── chatbot_interface.py   # Chat interfaces
├── demo_chat.py          # Demo script
├── models/               # Trained models
├── logs/                 # Training logs
├── plots/                # Training plots
├── dataset/              # Dataset files
├── requirements.txt      # Dependencies
├── setup.sh             # Setup script
├── run.sh               # Run script
└── README.md            # This file
```

## 🛠️ Dependencies

### Core Requirements
- Python 3.11+ (recommended)
- TensorFlow 2.20+
- NumPy, Pandas, Scikit-learn
- NLTK, Matplotlib, Seaborn
- Streamlit (web interface)

### Installation
```bash
pip install -r requirements.txt
```

## 📊 Usage Examples

### 1. Full Pipeline
```bash
python main.py full
```

### 2. Individual Steps
```bash
# Preprocessing
python main.py preprocess

# Data preparation  
python main.py prepare

# Training
python main.py train

# Chat
python main.py chat
```

### 3. Hybrid Approach
```bash
# Train intent classifier
python main.py intent-train

# Demo hybrid chatbot
python main.py intent-chat

# Or use hybrid chatbot directly
python hybrid_chatbot.py
```

### 4. Web Interface (Recommended)
```bash
# Modern Streamlit interface
python main.py streamlit

# Or directly
streamlit run streamlit_app.py
```

### 5. Quick Demo
```bash
# Non-interactive demo
python demo_chat.py

# Hybrid chatbot demo
python hybrid_chatbot.py
```

## 🎯 Optimizations Applied

### Data Enhancement
- **Enhanced Responses**: Context-based responses
- **Greeting Variations**: Multiple greeting patterns
- **Sentiment Integration**: Positive/negative sentiment data
- **Entity Recognition**: Named entity processing

### Model Optimization
- **Architecture**: Deeper networks with more parameters
- **Regularization**: Dropout and recurrent dropout
- **Training**: Early stopping, learning rate reduction
- **Generation**: Top-k sampling, temperature control

### Performance Improvements
- **Dataset Size**: 11x more training data
- **Vocabulary**: 2.6x larger vocabulary
- **Response Quality**: More coherent and contextual
- **Training Speed**: Optimized batch processing

## 🚨 Troubleshooting

### Common Issues
1. **Python Version**: Use Python 3.11 (not 3.13)
2. **Dependencies**: Install with `pip install -r requirements.txt`
3. **Memory**: Ensure sufficient RAM for training
4. **GPU**: Optional, CPU training works fine

### Error Solutions
- **Import Errors**: Check virtual environment activation
- **Memory Errors**: Reduce batch size in training
- **Model Loading**: Ensure models are trained first

## 📝 License

This project is for educational purposes. Please ensure you have the right to use the dataset.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs in `logs/` directory
3. Ensure all dependencies are installed
4. Verify Python version compatibility

---

**Note**: This chatbot uses RNN architecture optimized for Indonesian language with comprehensive dataset integration including sentiment analysis and entity recognition.