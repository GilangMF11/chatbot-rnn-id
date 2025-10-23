"""
Data preparation untuk training RNN chatbot
Termasuk tokenization, sequence creation, dan data splitting
"""

import json
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pickle

class DataPreparator:
    def __init__(self, max_vocab_size: int = 10000, max_sequence_length: int = 50):
        """
        Inisialisasi data preparator
        
        Args:
            max_vocab_size: Maksimal ukuran vocabulary
            max_sequence_length: Maksimal panjang sequence
        """
        self.max_vocab_size = max_vocab_size
        self.max_sequence_length = max_sequence_length
        self.tokenizer = Tokenizer(
            num_words=max_vocab_size,
            oov_token='<OOV>',
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        )
        self.vocab_size = 0
        self.word_to_index = {}
        self.index_to_word = {}
        
    def load_training_pairs(self, file_path: str) -> List[Tuple[str, str]]:
        """
        Load training pairs dari file JSON
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                training_pairs = json.load(f)
            return training_pairs
        except FileNotFoundError:
            print(f"Warning: {file_path} not found")
            return []
    
    def create_vocabulary(self, texts: List[str]) -> None:
        """
        Membuat vocabulary dari semua teks
        """
        # Fit tokenizer pada semua teks
        self.tokenizer.fit_on_texts(texts)
        
        # Update vocabulary info
        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.word_to_index = self.tokenizer.word_index
        self.index_to_word = {v: k for k, v in self.word_to_index.items()}
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Most common words: {list(self.word_to_index.keys())[:10]}")
    
    def text_to_sequences(self, texts: List[str]) -> np.ndarray:
        """
        Convert teks ke sequences
        """
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(
            sequences, 
            maxlen=self.max_sequence_length, 
            padding='post',
            truncating='post'
        )
        return padded_sequences
    
    def prepare_sequences(self, input_texts: List[str], output_texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Siapkan input dan output sequences untuk training
        """
        # Create vocabulary dari semua teks
        all_texts = input_texts + output_texts
        self.create_vocabulary(all_texts)
        
        # Convert ke sequences
        input_sequences = self.text_to_sequences(input_texts)
        output_sequences = self.text_to_sequences(output_texts)
        
        return input_sequences, output_sequences
    
    def create_sequences_for_language_model(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Membuat sequences untuk language modeling
        """
        # Create vocabulary
        self.create_vocabulary(texts)
        
        # Convert semua teks ke sequences
        all_sequences = self.text_to_sequences(texts)
        
        # Create input-output pairs untuk language modeling
        X, y = [], []
        
        for sequence in all_sequences:
            for i in range(1, len(sequence)):
                if sequence[i] != 0:  # Skip padding tokens
                    X.append(sequence[:i])
                    y.append(sequence[i])
        
        # Pad input sequences
        X_padded = pad_sequences(X, maxlen=self.max_sequence_length, padding='post')
        
        return X_padded, np.array(y)
    
    def split_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Split data menjadi train dan test
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def save_tokenizer(self, file_path: str) -> None:
        """
        Simpan tokenizer untuk digunakan nanti
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
    
    def load_tokenizer(self, file_path: str) -> None:
        """
        Load tokenizer yang sudah disimpan
        """
        with open(file_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.word_to_index = self.tokenizer.word_index
        self.index_to_word = {v: k for k, v in self.word_to_index.items()}
    
    def decode_sequence(self, sequence: np.ndarray) -> str:
        """
        Decode sequence kembali ke teks
        """
        words = []
        for idx in sequence:
            if idx == 0:  # Padding token
                break
            if idx in self.index_to_word:
                words.append(self.index_to_word[idx])
            else:
                words.append('<UNK>')
        return ' '.join(words)
    
    def get_sequence_lengths(self, sequences: np.ndarray) -> np.ndarray:
        """
        Hitung panjang sequence (non-padding)
        """
        lengths = []
        for seq in sequences:
            length = np.sum(seq != 0)
            lengths.append(length)
        return np.array(lengths)

def create_chatbot_dataset(training_pairs: List[Tuple[str, str]]) -> Dict:
    """
    Membuat dataset khusus untuk chatbot
    """
    input_texts = [pair[0] for pair in training_pairs]
    output_texts = [pair[1] for pair in training_pairs]
    
    # Buat vocabulary dari semua teks
    all_texts = input_texts + output_texts
    
    # Analisis dataset
    print(f"Total training pairs: {len(training_pairs)}")
    print(f"Average input length: {np.mean([len(text.split()) for text in input_texts]):.2f}")
    print(f"Average output length: {np.mean([len(text.split()) for text in output_texts]):.2f}")
    
    # Cari panjang maksimal
    max_input_length = max([len(text.split()) for text in input_texts])
    max_output_length = max([len(text.split()) for text in output_texts])
    
    print(f"Max input length: {max_input_length}")
    print(f"Max output length: {max_output_length}")
    
    return {
        'input_texts': input_texts,
        'output_texts': output_texts,
        'all_texts': all_texts,
        'max_input_length': max_input_length,
        'max_output_length': max_output_length
    }

def analyze_dataset_statistics(dataset: Dict) -> None:
    """
    Analisis statistik dataset
    """
    input_texts = dataset['input_texts']
    output_texts = dataset['output_texts']
    
    print("\n=== Dataset Statistics ===")
    print(f"Number of input texts: {len(input_texts)}")
    print(f"Number of output texts: {len(output_texts)}")
    
    # Input statistics
    input_lengths = [len(text.split()) for text in input_texts]
    print(f"\nInput text statistics:")
    print(f"  Mean length: {np.mean(input_lengths):.2f}")
    print(f"  Median length: {np.median(input_lengths):.2f}")
    print(f"  Max length: {np.max(input_lengths)}")
    print(f"  Min length: {np.min(input_lengths)}")
    
    # Output statistics
    output_lengths = [len(text.split()) for text in output_texts]
    print(f"\nOutput text statistics:")
    print(f"  Mean length: {np.mean(output_lengths):.2f}")
    print(f"  Median length: {np.median(output_lengths):.2f}")
    print(f"  Max length: {np.max(output_lengths)}")
    print(f"  Min length: {np.min(output_lengths)}")
    
    # Word frequency analysis
    all_words = []
    for text in input_texts + output_texts:
        all_words.extend(text.split())
    
    from collections import Counter
    word_freq = Counter(all_words)
    print(f"\nMost common words:")
    for word, freq in word_freq.most_common(10):
        print(f"  {word}: {freq}")

if __name__ == "__main__":
    # Load training pairs
    print("Loading training pairs...")
    training_pairs = []
    try:
        with open('training_pairs.json', 'r', encoding='utf-8') as f:
            training_pairs = json.load(f)
    except FileNotFoundError:
        print("training_pairs.json not found. Please run preprocessing.py first.")
        exit(1)
    
    # Create dataset
    dataset = create_chatbot_dataset(training_pairs)
    
    # Analyze dataset
    analyze_dataset_statistics(dataset)
    
    # Prepare data
    preparator = DataPreparator(max_vocab_size=8000, max_sequence_length=30)
    
    # Create sequences untuk language modeling
    all_texts = dataset['all_texts']
    X, y = preparator.create_sequences_for_language_model(all_texts)
    
    print(f"\nPrepared sequences:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = preparator.split_data(X, y, test_size=0.2)
    
    print(f"\nData split:")
    print(f"Train: {X_train.shape[0]} samples")
    print(f"Test: {X_test.shape[0]} samples")
    
    # Save prepared data
    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
    
    # Save tokenizer
    preparator.save_tokenizer('tokenizer.pkl')
    
    print("\nData preparation completed!")
    print("Files saved: X_train.npy, X_test.npy, y_train.npy, y_test.npy, tokenizer.pkl")
