"""
Model RNN untuk chatbot bahasa Indonesia
Menggunakan SimpleRNN dengan attention mechanism
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    SimpleRNN, Dense, Embedding, Dropout, 
    Input, Bidirectional, Attention, 
    Concatenate, TimeDistributed, GlobalMaxPooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import pickle
from typing import Tuple, Dict, List

class IndonesianChatbotRNN:
    def __init__(self, vocab_size: int, max_sequence_length: int, embedding_dim: int = 128):
        """
        Inisialisasi model RNN chatbot
        
        Args:
            vocab_size: Ukuran vocabulary
            max_sequence_length: Maksimal panjang sequence
            embedding_dim: Dimensi embedding
        """
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.model = None
        self.tokenizer = None
        
    def build_simple_rnn_model(self, rnn_units: int = 256, dropout_rate: float = 0.3) -> Model:
        """
        Membuat model RNN yang dioptimalkan untuk chatbot
        """
        model = Sequential([
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim * 2,  # Double embedding size
                input_length=self.max_sequence_length,
                mask_zero=True
            ),
            SimpleRNN(rnn_units, return_sequences=True, dropout=dropout_rate, recurrent_dropout=dropout_rate),
            SimpleRNN(rnn_units, dropout=dropout_rate, recurrent_dropout=dropout_rate),
            Dropout(dropout_rate),
            Dense(512, activation='relu'),
            Dropout(dropout_rate),
            Dense(256, activation='relu'),
            Dropout(dropout_rate),
            Dense(self.vocab_size, activation='softmax')
        ])
        
        # Optimizer dengan learning rate yang lebih rendah
        model.compile(
            optimizer=Adam(learning_rate=0.0005),  # Lower learning rate
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_bidirectional_rnn_model(self, rnn_units: int = 128, dropout_rate: float = 0.2) -> Model:
        """
        Membuat model Bidirectional RNN
        """
        model = Sequential([
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_sequence_length,
                mask_zero=True
            ),
            Bidirectional(SimpleRNN(rnn_units, return_sequences=True, dropout=dropout_rate)),
            Bidirectional(SimpleRNN(rnn_units, dropout=dropout_rate)),
            Dropout(dropout_rate),
            Dense(512, activation='relu'),
            Dropout(dropout_rate),
            Dense(256, activation='relu'),
            Dropout(dropout_rate),
            Dense(self.vocab_size, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_attention_rnn_model(self, rnn_units: int = 128, dropout_rate: float = 0.2) -> Model:
        """
        Membuat model RNN dengan attention mechanism
        """
        # Input layer
        inputs = Input(shape=(self.max_sequence_length,))
        
        # Embedding layer
        embedding = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.max_sequence_length,
            mask_zero=True
        )(inputs)
        
        # RNN layers
        rnn1 = SimpleRNN(rnn_units, return_sequences=True, dropout=dropout_rate)(embedding)
        rnn2 = SimpleRNN(rnn_units, return_sequences=True, dropout=dropout_rate)(rnn1)
        
        # Attention mechanism
        attention = Dense(1, activation='tanh')(rnn2)
        attention = tf.nn.softmax(attention, axis=1)
        context_vector = tf.reduce_sum(rnn2 * attention, axis=1)
        
        # Dense layers
        dense1 = Dense(512, activation='relu')(context_vector)
        dropout1 = Dropout(dropout_rate)(dense1)
        dense2 = Dense(256, activation='relu')(dropout1)
        dropout2 = Dropout(dropout_rate)(dense2)
        outputs = Dense(self.vocab_size, activation='softmax')(dropout2)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_chatbot_model(self, rnn_units: int = 128, dropout_rate: float = 0.2) -> Model:
        """
        Membuat model khusus untuk chatbot dengan encoder-decoder architecture menggunakan RNN
        """
        # Encoder
        encoder_inputs = Input(shape=(self.max_sequence_length,))
        encoder_embedding = Embedding(self.vocab_size, self.embedding_dim, mask_zero=True)(encoder_inputs)
        encoder_rnn = SimpleRNN(rnn_units, return_state=True, dropout=dropout_rate)
        encoder_outputs, state_h = encoder_rnn(encoder_embedding)
        encoder_states = [state_h]
        
        # Decoder
        decoder_inputs = Input(shape=(self.max_sequence_length,))
        decoder_embedding = Embedding(self.vocab_size, self.embedding_dim, mask_zero=True)(decoder_inputs)
        decoder_rnn = SimpleRNN(rnn_units, return_sequences=True, return_state=True, dropout=dropout_rate)
        decoder_outputs, _ = decoder_rnn(decoder_embedding, initial_state=encoder_states)
        
        # Attention layer
        attention = Dense(1, activation='tanh')(decoder_outputs)
        attention_weights = tf.nn.softmax(attention, axis=1)
        context_vector = tf.reduce_sum(decoder_outputs * attention_weights, axis=1)
        
        # Output layer
        dense1 = Dense(512, activation='relu')(context_vector)
        dropout1 = Dropout(dropout_rate)(dense1)
        dense2 = Dense(256, activation='relu')(dropout1)
        dropout2 = Dropout(dropout_rate)(dense2)
        outputs = Dense(self.vocab_size, activation='softmax')(dropout2)
        
        model = Model([encoder_inputs, decoder_inputs], outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: np.ndarray, y_val: np.ndarray, 
                   model_type: str = 'simple_rnn', 
                   epochs: int = 100, batch_size: int = 16) -> Dict:
        """
        Training model RNN
        """
        # Pilih model berdasarkan tipe
        if model_type == 'simple_rnn':
            self.model = self.build_simple_rnn_model()
        elif model_type == 'bidirectional_rnn':
            self.model = self.build_bidirectional_rnn_model()
        elif model_type == 'attention_rnn':
            self.model = self.build_attention_rnn_model()
        elif model_type == 'chatbot':
            self.model = self.build_chatbot_model()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Callbacks yang dioptimalkan untuk chatbot
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=1e-6),
            ModelCheckpoint(f'best_model_{model_type}.h5', monitor='val_loss', save_best_only=True)
        ]
        
        # Training
        print(f"Training {model_type} model...")
        print(f"Model summary:")
        self.model.summary()
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history.history
    
    def generate_text(self, seed_text: str, max_length: int = 30, temperature: float = 0.7) -> str:
        """
        Generate teks yang dioptimalkan untuk chatbot
        """
        if self.model is None:
            raise ValueError("Model belum dilatih!")
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer belum di-load!")
        
        # Convert seed text ke sequence
        seed_sequence = self.tokenizer.texts_to_sequences([seed_text])
        seed_sequence = tf.keras.preprocessing.sequence.pad_sequences(
            seed_sequence, maxlen=self.max_sequence_length, padding='post'
        )
        
        generated_text = seed_text
        current_sequence = seed_sequence[0]
        
        # Limit generation untuk response yang lebih fokus
        for _ in range(min(max_length, 20)):
            # Predict next word
            prediction = self.model.predict(current_sequence.reshape(1, -1), verbose=0)
            
            # Apply temperature (lower = more focused)
            prediction = prediction / temperature
            prediction = tf.nn.softmax(prediction)
            
            # Normalize probabilities
            probs = prediction[0].numpy()
            probs = probs / np.sum(probs)
            
            # Top-k sampling untuk response yang lebih koheren
            top_k = 10
            top_indices = np.argsort(probs)[-top_k:]
            top_probs = probs[top_indices]
            top_probs = top_probs / np.sum(top_probs)
            
            next_word_idx = np.random.choice(top_indices, p=top_probs)
            
            # Check if it's end token or padding
            if next_word_idx == 0 or next_word_idx >= self.vocab_size:
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
    
    def save_model(self, filepath: str) -> None:
        """
        Simpan model dan tokenizer
        """
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load model yang sudah disimpan
        """
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def set_tokenizer(self, tokenizer) -> None:
        """
        Set tokenizer
        """
        self.tokenizer = tokenizer
        print("Tokenizer set successfully")
    
    def load_tokenizer(self, filepath: str) -> None:
        """
        Load tokenizer
        """
        with open(filepath, 'rb') as f:
            self.tokenizer = pickle.load(f)
        print(f"Tokenizer loaded from {filepath}")
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluasi model
        """
        if self.model is None:
            raise ValueError("Model belum dilatih!")
        
        # Evaluate model
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Predictions
        predictions = self.model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Calculate additional metrics
        from sklearn.metrics import classification_report, confusion_matrix
        
        # Get top-k accuracy
        top_5_accuracy = tf.keras.metrics.sparse_top_k_categorical_accuracy(y_test, predictions, k=5)
        top_5_accuracy = np.mean(top_5_accuracy)
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'top_5_accuracy': top_5_accuracy,
            'predictions': predicted_classes
        }
    
    def plot_training_history(self, history: Dict, save_path: str = None) -> None:
        """
        Plot training history
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(history['loss'], label='Training Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot accuracy
        ax2.plot(history['accuracy'], label='Training Accuracy')
        ax2.plot(history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

def compare_models(X_train: np.ndarray, y_train: np.ndarray, 
                  X_val: np.ndarray, y_val: np.ndarray,
                  vocab_size: int, max_sequence_length: int) -> Dict:
    """
    Bandingkan performa berbagai model RNN
    """
    model_types = ['simple_lstm', 'bidirectional_lstm', 'attention_lstm']
    results = {}
    
    for model_type in model_types:
        print(f"\n=== Training {model_type.upper()} ===")
        
        # Create model
        chatbot = IndonesianChatbotRNN(vocab_size, max_sequence_length)
        
        # Train model
        history = chatbot.train_model(
            X_train, y_train, X_val, y_val, 
            model_type=model_type, epochs=20, batch_size=32
        )
        
        # Evaluate
        eval_results = chatbot.evaluate_model(X_val, y_val)
        
        results[model_type] = {
            'history': history,
            'evaluation': eval_results,
            'model': chatbot
        }
        
        print(f"{model_type} - Accuracy: {eval_results['accuracy']:.4f}")
    
    return results

if __name__ == "__main__":
    # Load prepared data
    print("Loading prepared data...")
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')
    
    # Load tokenizer
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    vocab_size = len(tokenizer.word_index) + 1
    max_sequence_length = X_train.shape[1]
    
    print(f"Data shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
    print(f"Vocab size: {vocab_size}")
    print(f"Max sequence length: {max_sequence_length}")
    
    # Create and train model
    chatbot = IndonesianChatbotRNN(vocab_size, max_sequence_length)
    chatbot.tokenizer = tokenizer
    
    # Train simple LSTM model
    print("\nTraining simple LSTM model...")
    history = chatbot.train_model(
        X_train, y_train, X_test, y_test,
        model_type='simple_lstm', epochs=30, batch_size=64
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    eval_results = chatbot.evaluate_model(X_test, y_test)
    print(f"Test Accuracy: {eval_results['accuracy']:.4f}")
    print(f"Test Loss: {eval_results['loss']:.4f}")
    print(f"Top-5 Accuracy: {eval_results['top_5_accuracy']:.4f}")
    
    # Save model
    chatbot.save_model('indonesian_chatbot_model.h5')
    
    # Test text generation
    print("\nTesting text generation...")
    test_seeds = ["halo", "apa kabar", "terima kasih", "selamat pagi"]
    
    for seed in test_seeds:
        generated = chatbot.generate_text(seed, max_length=20, temperature=0.8)
        print(f"Seed: '{seed}' -> Generated: '{generated}'")
    
    print("\nTraining completed!")
