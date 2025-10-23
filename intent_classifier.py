"""
Intent Classification untuk Chatbot Bahasa Indonesia
Menggunakan corpus dialog yang terstruktur
"""

import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os
from typing import List, Dict, Tuple

class IntentClassifier:
    """
    Classifier untuk intent detection berdasarkan corpus dialog
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words=None,  # Tidak ada stopwords untuk bahasa Indonesia
            lowercase=True
        )
        self.classifier = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        self.intent_responses = {}
        self.is_trained = False
        
    def load_corpus_data(self) -> Tuple[List[str], List[str]]:
        """
        Load data dari corpus untuk intent classification
        """
        corpus_files = [
            'dataset/corpus/id/dialog.json',
            'dataset/corpus/id/agent.json', 
            'dataset/corpus/id/user.json',
            'dataset/corpus/id/motivasi.json',
            'dataset/corpus/id/None.json'
        ]
        
        texts = []
        intents = []
        responses = []
        
        for file_path in corpus_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                for item in data:
                    intent = item.get('intent', '')
                    utterances = item.get('utterances', [])
                    answers = item.get('answers', [])
                    
                    # Simpan mapping intent -> responses
                    if intent not in self.intent_responses:
                        self.intent_responses[intent] = answers
                    
                    # Buat training data
                    for utterance in utterances:
                        texts.append(utterance)
                        intents.append(intent)
                        
            except FileNotFoundError:
                print(f"Warning: {file_path} not found")
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON in {file_path}")
        
        return texts, intents
    
    def train(self):
        """
        Train intent classifier
        """
        print("🔄 Loading corpus data...")
        texts, intents = self.load_corpus_data()
        
        if not texts:
            print("❌ No data found!")
            return False
            
        print(f"📊 Loaded {len(texts)} utterances with {len(set(intents))} intents")
        
        # Vectorize texts
        print("🔄 Vectorizing texts...")
        X = self.vectorizer.fit_transform(texts)
        y = np.array(intents)
        
        # Split data (without stratify for small classes)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train classifier
        print("🔄 Training intent classifier...")
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Intent Classification Accuracy: {accuracy:.3f}")
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        self.is_trained = True
        return True
    
    def predict_intent(self, text: str) -> str:
        """
        Predict intent dari input text
        """
        if not self.is_trained:
            return "unknown"
            
        # Vectorize input
        X = self.vectorizer.transform([text])
        
        # Predict
        intent = self.classifier.predict(X)[0]
        confidence = self.classifier.predict_proba(X).max()
        
        return intent if confidence > 0.3 else "unknown"
    
    def get_response(self, text: str) -> str:
        """
        Get response berdasarkan intent
        """
        intent = self.predict_intent(text)
        
        if intent == "unknown" or intent not in self.intent_responses:
            return "Maaf, saya tidak mengerti. Bisa dijelaskan lebih detail?"
        
        # Pilih response secara random dari available responses
        responses = self.intent_responses[intent]
        return np.random.choice(responses)
    
    def save_model(self, filepath: str = "models/intent_classifier.pkl"):
        """
        Save trained model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'intent_responses': self.intent_responses,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Intent classifier saved to {filepath}")
    
    def load_model(self, filepath: str = "models/intent_classifier.pkl"):
        """
        Load trained model
        """
        if not os.path.exists(filepath):
            print(f"❌ Model not found: {filepath}")
            return False
            
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']
        self.intent_responses = model_data['intent_responses']
        self.is_trained = model_data['is_trained']
        
        print(f"✅ Intent classifier loaded from {filepath}")
        return True

def train_intent_classifier():
    """
    Train intent classifier
    """
    classifier = IntentClassifier()
    
    if classifier.train():
        classifier.save_model()
        return classifier
    else:
        return None

if __name__ == "__main__":
    print("🤖 Training Intent Classifier for Indonesian Chatbot")
    print("=" * 60)
    
    classifier = train_intent_classifier()
    
    if classifier:
        print("\n💬 Testing Intent Classification:")
        print("-" * 40)
        
        test_inputs = [
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
        
        for text in test_inputs:
            intent = classifier.predict_intent(text)
            response = classifier.get_response(text)
            print(f"👤 '{text}' → Intent: {intent}")
            print(f"🤖 Response: {response}")
            print()
    else:
        print("❌ Training failed!")
