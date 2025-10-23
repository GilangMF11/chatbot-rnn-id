# 📖 Panduan Penggunaan Chatbot RNN Bahasa Indonesia

## 🚀 Quick Start

### 1. Setup (Hanya sekali)
```bash
# Jalankan setup script
./setup.sh

# Atau manual:
python3.11 -m venv chatbot_env
source chatbot_env/bin/activate
pip install -r requirements.txt
```

### 2. Jalankan Program
```bash
# Menggunakan run script (recommended)
./run.sh

# Atau manual:
source chatbot_env/bin/activate
python main.py [command]
```

## 📋 Commands

| Command | Deskripsi |
|---------|-----------|
| `python main.py check` | Cek dependencies |
| `python main.py preprocess` | Jalankan preprocessing |
| `python main.py prepare` | Siapkan data training |
| `python main.py train` | Training model RNN |
| `python main.py chat` | Chat interface (console) |
| `python main.py streamlit` | Chat interface (web) |
| `python main.py full` | Jalankan semua pipeline |

## 🔧 Troubleshooting

### Error: "externally-managed-environment"
```bash
# Gunakan virtual environment
python3.11 -m venv chatbot_env
source chatbot_env/bin/activate
pip install -r requirements.txt
```

### Error: "TensorFlow not found"
```bash
# Pastikan menggunakan Python 3.11
python3.11 --version
# Install TensorFlow
pip install tensorflow>=2.20.0
```

### Error: "EOF when reading a line"
- Ini adalah masalah environment yang tidak bisa diatasi dengan mudah
- Gunakan Python 3.11 dan virtual environment yang bersih

## 📁 Struktur File

```
chatbot-rnn-id/
├── main.py                 # Script utama
├── preprocessing.py        # Preprocessing data
├── data_preparation.py     # Persiapan data
├── rnn_model.py           # Model RNN
├── training_pipeline.py   # Pipeline training
├── chatbot_interface.py   # Interface chatbot
├── requirements.txt       # Dependencies
├── setup.sh              # Setup script
├── run.sh                # Run script
├── README.md             # Dokumentasi utama
├── USAGE.md              # Panduan penggunaan
└── dataset/              # Dataset
    ├── corpus/id/         # Data corpus
    ├── entities/          # Entities
    ├── normalization/     # Normalisasi
    └── sentiment/         # Sentiment data
```

## 🎯 Workflow Lengkap

1. **Setup**: `./setup.sh`
2. **Check**: `python main.py check`
3. **Preprocess**: `python main.py preprocess`
4. **Prepare**: `python main.py prepare`
5. **Train**: `python main.py train`
6. **Chat**: `python main.py chat` atau `python main.py streamlit`

## 💡 Tips

- Gunakan Python 3.11 untuk kompatibilitas terbaik
- Selalu aktifkan virtual environment sebelum menjalankan
- Untuk development, gunakan `python main.py full` untuk menjalankan semua pipeline
- Untuk production, jalankan step by step untuk monitoring yang lebih baik
