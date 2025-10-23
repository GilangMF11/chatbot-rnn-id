# 🌐 Ngrok Setup untuk Chatbot RNN Bahasa Indonesia

Panduan lengkap untuk menjalankan chatbot dengan ngrok tunneling agar bisa diakses dari internet.

## 📋 Prerequisites

### 1. Install Ngrok
```bash
# Download dari https://ngrok.com/download
# Atau install via brew (macOS)
brew install ngrok

# Atau install via pip
pip install pyngrok
```

### 2. Setup Ngrok Account
1. Daftar di [ngrok.com](https://ngrok.com)
2. Dapatkan authtoken dari dashboard
3. Setup authtoken:
```bash
ngrok config add-authtoken YOUR_AUTHTOKEN
```

### 3. Virtual Environment
```bash
source chatbot_env/bin/activate
```

## 🚀 Cara Menjalankan

### Method 1: Script Python (Recommended)
```bash
# Pastikan virtual environment aktif
source chatbot_env/bin/activate

# Jalankan script ngrok
python ngrok_simple.py
```

### Method 2: Script Bash
```bash
# Pastikan virtual environment aktif
source chatbot_env/bin/activate

# Jalankan script bash
./ngrok_setup.sh
```

### Method 3: Manual
```bash
# Terminal 1: Start Streamlit
source chatbot_env/bin/activate
python -m streamlit run streamlit_demo.py --server.port 8501

# Terminal 2: Start ngrok
ngrok http 8501
```

## 📊 Output yang Diharapkan

```
🤖 Chatbot RNN Bahasa Indonesia - Ngrok Tunnel
==================================================
🔍 Checking requirements...
✅ Ngrok found
✅ All model files found!
✅ All requirements met!
🚀 Starting Streamlit app...
✅ Streamlit started on port 8501
🌐 Starting ngrok tunnel...
✅ Ngrok tunnel started
🔍 Getting tunnel URL...

🎉 Chatbot is now accessible via internet!
🌐 Public URL: https://abc123.ngrok.io
🔗 Local URL: http://localhost:8501
📊 Ngrok Dashboard: http://localhost:4040

💡 Share the Public URL with others!
💡 Press Ctrl+C to stop
```

## 🔧 Troubleshooting

### Error: Ngrok not found
```bash
# Install ngrok
brew install ngrok
# atau download dari https://ngrok.com/download
```

### Error: Virtual environment not active
```bash
source chatbot_env/bin/activate
```

### Error: Missing model files
```bash
# Run training first
python main.py train
python main.py intent-train
```

### Error: Port already in use
```bash
# Kill existing processes
pkill -f streamlit
pkill -f ngrok
```

## 🌐 Mengakses Chatbot

### Local Access
- **URL**: http://localhost:8501
- **Dashboard**: http://localhost:4040

### Public Access
- **URL**: https://abc123.ngrok.io (akan berbeda setiap kali)
- **Share**: Kirim URL ini ke orang lain untuk akses

## 📱 Fitur yang Tersedia

### 1. Model Selection
- **Hybrid (Intent + RNN)**: Kombinasi intent classification dan RNN
- **Intent Only**: Hanya intent classification
- **RNN Only**: Hanya RNN text generation

### 2. Chat Interface
- **Modern UI**: Chat bubbles dengan gradient styling
- **Real-time Chat**: Interface chat yang responsif
- **Statistics**: Dashboard statistik percakapan
- **Quick Test**: Test dengan input yang sudah ditentukan

### 3. Model Management
- **Load Models**: Tombol untuk load chatbot models
- **Model Status**: Indikator status model
- **Error Handling**: Pesan error yang informatif

## 🔒 Security Notes

1. **Public Access**: URL ngrok bersifat publik, hati-hati dengan data sensitif
2. **Authentication**: Pertimbangkan menambah authentication jika diperlukan
3. **Rate Limiting**: Ngrok free tier memiliki batasan
4. **HTTPS**: Ngrok menyediakan HTTPS secara default

## 📈 Performance Tips

1. **Model Loading**: Load models sekali di awal untuk performa terbaik
2. **Caching**: Streamlit akan cache model yang sudah di-load
3. **Memory**: Pastikan RAM cukup untuk model RNN
4. **Network**: Koneksi internet yang stabil untuk ngrok

## 🛠️ Advanced Configuration

### Custom Domain (Ngrok Pro)
```bash
ngrok http 8501 --hostname=your-custom-domain.ngrok.io
```

### Custom Port
```bash
# Edit ngrok_simple.py atau ngrok_setup.sh
# Ubah port 8501 ke port yang diinginkan
```

### Multiple Tunnels
```bash
# Terminal 1
ngrok http 8501

# Terminal 2  
ngrok http 8502
```

## 📞 Support

Jika mengalami masalah:
1. Check log error di terminal
2. Pastikan semua requirements terpenuhi
3. Restart virtual environment
4. Check ngrok dashboard di http://localhost:4040

---

**Selamat menggunakan chatbot dengan ngrok tunneling!** 🎉
