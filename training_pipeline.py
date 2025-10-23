"""
Pipeline training lengkap untuk chatbot RNN bahasa Indonesia
Termasuk training, evaluation, dan model comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json
import pickle
import os
from datetime import datetime
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from rnn_model import IndonesianChatbotRNN
from data_preparation import DataPreparator

class TrainingPipeline:
    def __init__(self, model_save_dir: str = 'models'):
        """
        Inisialisasi training pipeline
        
        Args:
            model_save_dir: Direktori untuk menyimpan model
        """
        self.model_save_dir = model_save_dir
        self.results = {}
        
        # Buat direktori jika belum ada
        os.makedirs(model_save_dir, exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data yang sudah diproses
        """
        print("Loading prepared data...")
        
        X_train = np.load('X_train.npy')
        X_test = np.load('X_test.npy')
        y_train = np.load('y_train.npy')
        y_test = np.load('y_test.npy')
        
        print(f"Data shapes:")
        print(f"X_train: {X_train.shape}")
        print(f"X_test: {X_test.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"y_test: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def load_tokenizer(self) -> Tuple:
        """
        Load tokenizer dan info vocabulary
        """
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        
        vocab_size = len(tokenizer.word_index) + 1
        max_sequence_length = X_train.shape[1] if 'X_train' in globals() else 30
        
        return tokenizer, vocab_size, max_sequence_length
    
    def train_single_model(self, model_type: str, X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray, 
                          vocab_size: int, max_sequence_length: int,
                          epochs: int = 50, batch_size: int = 32) -> Dict:
        """
        Train single model
        """
        print(f"\n{'='*50}")
        print(f"Training {model_type.upper()} Model")
        print(f"{'='*50}")
        
        # Create model
        chatbot = IndonesianChatbotRNN(vocab_size, max_sequence_length)
        
        # Set tokenizer
        tokenizer, _, _ = self.load_tokenizer()
        chatbot.set_tokenizer(tokenizer)
        
        # Train model
        start_time = datetime.now()
        history = chatbot.train_model(
            X_train, y_train, X_test, y_test,
            model_type=model_type, epochs=epochs, batch_size=batch_size
        )
        training_time = datetime.now() - start_time
        
        # Evaluate model
        eval_results = chatbot.evaluate_model(X_test, y_test)
        
        # Save model
        model_path = os.path.join(self.model_save_dir, f'{model_type}_model.h5')
        chatbot.save_model(model_path)
        
        # Save tokenizer
        tokenizer_path = os.path.join(self.model_save_dir, f'{model_type}_tokenizer.pkl')
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(chatbot.tokenizer, f)
        
        # Plot training history
        self.plot_training_history(history, model_type)
        
        # Test text generation
        test_results = self.test_text_generation(chatbot, model_type)
        
        return {
            'model_type': model_type,
            'history': history,
            'evaluation': eval_results,
            'training_time': str(training_time),
            'model_path': model_path,
            'tokenizer_path': tokenizer_path,
            'test_results': test_results
        }
    
    def train_multiple_models(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_test: np.ndarray, y_test: np.ndarray,
                             vocab_size: int, max_sequence_length: int,
                             model_types: List[str] = None, epochs: int = 30) -> Dict:
        """
        Train multiple models dan bandingkan performanya
        """
        if model_types is None:
            model_types = ['simple_rnn', 'bidirectional_rnn', 'attention_rnn']
        
        print(f"Training {len(model_types)} different models...")
        
        results = {}
        
        for model_type in model_types:
            try:
                result = self.train_single_model(
                    model_type, X_train, y_train, X_test, y_test,
                    vocab_size, max_sequence_length, epochs=epochs
                )
                results[model_type] = result
                
                print(f"\n{model_type} completed:")
                print(f"  Accuracy: {result['evaluation']['accuracy']:.4f}")
                print(f"  Loss: {result['evaluation']['loss']:.4f}")
                print(f"  Training time: {result['training_time']}")
                
            except Exception as e:
                print(f"Error training {model_type}: {str(e)}")
                results[model_type] = {'error': str(e)}
        
        # Save results
        self.save_results(results)
        
        # Create comparison plots
        self.plot_model_comparison(results)
        
        return results
    
    def test_text_generation(self, chatbot: IndonesianChatbotRNN, model_type: str) -> Dict:
        """
        Test text generation dengan berbagai seed
        """
        test_seeds = [
            "halo", "apa kabar", "terima kasih", "selamat pagi",
            "bagaimana", "siapa", "kapan", "dimana", "mengapa"
        ]
        
        generation_results = {}
        
        print(f"\nTesting text generation for {model_type}:")
        
        for seed in test_seeds:
            try:
                # Generate dengan temperature berbeda
                generated_cold = chatbot.generate_text(seed, max_length=15, temperature=0.3)
                generated_warm = chatbot.generate_text(seed, max_length=15, temperature=0.8)
                generated_hot = chatbot.generate_text(seed, max_length=15, temperature=1.2)
                
                generation_results[seed] = {
                    'cold': generated_cold,
                    'warm': generated_warm,
                    'hot': generated_hot
                }
                
                print(f"  Seed: '{seed}'")
                print(f"    Cold (0.3): {generated_cold}")
                print(f"    Warm (0.8): {generated_warm}")
                print(f"    Hot (1.2): {generated_hot}")
                
            except Exception as e:
                print(f"  Error generating for '{seed}': {str(e)}")
                generation_results[seed] = {'error': str(e)}
        
        return generation_results
    
    def plot_training_history(self, history: Dict, model_type: str) -> None:
        """
        Plot training history untuk single model
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(history['loss'], label='Training Loss', linewidth=2)
        ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_title(f'{model_type.upper()} - Model Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        ax2.plot(history['accuracy'], label='Training Accuracy', linewidth=2)
        ax2.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax2.set_title(f'{model_type.upper()} - Model Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'plots/{model_type}_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, results: Dict) -> None:
        """
        Plot perbandingan performa berbagai model
        """
        # Extract metrics
        model_names = []
        accuracies = []
        losses = []
        training_times = []
        
        for model_type, result in results.items():
            if 'error' not in result:
                model_names.append(model_type.replace('_', ' ').title())
                accuracies.append(result['evaluation']['accuracy'])
                losses.append(result['evaluation']['loss'])
                
                # Parse training time
                time_str = result['training_time']
                if ':' in time_str:
                    parts = time_str.split(':')
                    if len(parts) == 3:  # HH:MM:SS
                        hours, minutes, seconds = parts
                        total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds.split('.')[0])
                        training_times.append(total_seconds)
                    else:
                        training_times.append(0)
                else:
                    training_times.append(0)
        
        if not model_names:
            print("No valid results to plot")
            return
        
        # Create comparison plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy comparison
        bars1 = ax1.bar(model_names, accuracies, color='skyblue', alpha=0.7)
        ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        for i, v in enumerate(accuracies):
            ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Loss comparison
        bars2 = ax2.bar(model_names, losses, color='lightcoral', alpha=0.7)
        ax2.set_title('Model Loss Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Loss')
        for i, v in enumerate(losses):
            ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Training time comparison
        bars3 = ax3.bar(model_names, training_times, color='lightgreen', alpha=0.7)
        ax3.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Time (seconds)')
        for i, v in enumerate(training_times):
            ax3.text(i, v + max(training_times) * 0.01, f'{v}s', ha='center', va='bottom')
        
        # Combined metrics
        x = np.arange(len(model_names))
        width = 0.35
        
        ax4.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.7)
        ax4_twin = ax4.twinx()
        ax4_twin.bar(x + width/2, losses, width, label='Loss', alpha=0.7, color='red')
        
        ax4.set_title('Accuracy vs Loss', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Accuracy')
        ax4_twin.set_ylabel('Loss')
        ax4.set_xticks(x)
        ax4.set_xticklabels(model_names)
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig('plots/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, results: Dict) -> None:
        """
        Simpan hasil training
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = f'logs/training_results_{timestamp}.json'
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for model_type, result in results.items():
            if 'error' not in result:
                # Convert numpy types to Python types
                evaluation = {}
                for key, value in result['evaluation'].items():
                    if hasattr(value, 'item') and value.size == 1:  # numpy scalar
                        evaluation[key] = value.item()
                    elif isinstance(value, (np.integer, np.floating)):
                        evaluation[key] = float(value)
                    elif isinstance(value, np.ndarray):
                        evaluation[key] = value.tolist()
                    else:
                        evaluation[key] = value
                
                serializable_results[model_type] = {
                    'model_type': result['model_type'],
                    'evaluation': evaluation,
                    'training_time': str(result['training_time']),
                    'model_path': result['model_path'],
                    'tokenizer_path': result['tokenizer_path']
                }
            else:
                serializable_results[model_type] = {'error': result['error']}
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        # Save summary
        summary_file = f'logs/training_summary_{timestamp}.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("TRAINING SUMMARY\n")
            f.write("="*50 + "\n\n")
            
            for model_type, result in results.items():
                if 'error' not in result:
                    f.write(f"{model_type.upper()}:\n")
                    f.write(f"  Accuracy: {result['evaluation']['accuracy']:.4f}\n")
                    f.write(f"  Loss: {result['evaluation']['loss']:.4f}\n")
                    f.write(f"  Training Time: {result['training_time']}\n")
                    f.write(f"  Model Path: {result['model_path']}\n\n")
                else:
                    f.write(f"{model_type.upper()}: ERROR - {result['error']}\n\n")
        
        print(f"Results saved to {results_file}")
        print(f"Summary saved to {summary_file}")
    
    def run_full_pipeline(self, epochs: int = 30) -> Dict:
        """
        Jalankan full training pipeline
        """
        print("Starting full training pipeline...")
        print("="*60)
        
        # Load data
        X_train, X_test, y_train, y_test = self.load_data()
        
        # Load tokenizer info
        tokenizer, vocab_size, max_sequence_length = self.load_tokenizer()
        
        print(f"\nDataset Info:")
        print(f"  Vocabulary size: {vocab_size}")
        print(f"  Max sequence length: {max_sequence_length}")
        print(f"  Training samples: {X_train.shape[0]}")
        print(f"  Test samples: {X_test.shape[0]}")
        
        # Train multiple models
        model_types = ['simple_rnn', 'bidirectional_rnn', 'attention_rnn']
        
        results = self.train_multiple_models(
            X_train, y_train, X_test, y_test,
            vocab_size, max_sequence_length,
            model_types=model_types, epochs=epochs
        )
        
        # Print final summary
        print("\n" + "="*60)
        print("FINAL RESULTS SUMMARY")
        print("="*60)
        
        best_accuracy = 0
        best_model = None
        
        for model_type, result in results.items():
            if 'error' not in result:
                accuracy = result['evaluation']['accuracy']
                print(f"{model_type.upper()}:")
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  Loss: {result['evaluation']['loss']:.4f}")
                print(f"  Training Time: {result['training_time']}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model_type
            else:
                print(f"{model_type.upper()}: ERROR - {result['error']}")
        
        if best_model:
            print(f"\nBest Model: {best_model.upper()} (Accuracy: {best_accuracy:.4f})")
            print(f"Model saved at: {results[best_model]['model_path']}")
        
        return results

if __name__ == "__main__":
    # Run full training pipeline
    pipeline = TrainingPipeline()
    results = pipeline.run_full_pipeline(epochs=20)
    
    print("\nTraining pipeline completed!")
    print("Check the 'models/', 'logs/', and 'plots/' directories for outputs.")
