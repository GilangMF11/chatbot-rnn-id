"""
Pipeline preprocessing untuk chatbot bahasa Indonesia
Menggunakan normalisasi, stopword removal, dan tokenization
"""

import json
import re
import string
import pandas as pd
from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict

class IndonesianTextPreprocessor:
    def __init__(self, normalization_file: str, stopword_file: str):
        """
        Inisialisasi preprocessor dengan file normalisasi dan stopword
        """
        self.normalization_dict = self._load_normalization(normalization_file)
        self.stopwords = self._load_stopwords(stopword_file)
        self.slang_dict = self._load_slang_dict()
        self.emoticon_sentiment = self._load_emoticon_sentiment()
        
    def _load_normalization(self, file_path: str) -> Dict[str, str]:
        """Load normalization dictionary"""
        normalization_dict = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if '\t' in line:
                        slang, formal = line.split('\t', 1)
                        normalization_dict[slang.lower()] = formal.lower()
        except FileNotFoundError:
            print(f"Warning: {file_path} not found")
        return normalization_dict
    
    def _load_stopwords(self, file_path: str) -> set:
        """Load stopwords list"""
        stopwords = set()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    stopwords.add(line.strip().lower())
        except FileNotFoundError:
            print(f"Warning: {file_path} not found")
        return stopwords
    
    def _load_slang_dict(self) -> Dict[str, str]:
        """Load slang dictionary from slangword.txt"""
        slang_dict = {}
        try:
            with open('dataset/notNormalized/slangword.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if ':' in line:
                        slang, formal = line.split(':', 1)
                        slang_dict[slang.lower()] = formal.lower()
        except FileNotFoundError:
            print("Warning: slangword.txt not found")
        return slang_dict
    
    def _load_emoticon_sentiment(self) -> Dict[str, str]:
        """Load emoticon sentiment mapping"""
        emoticon_dict = {}
        try:
            with open('dataset/notNormalized/emoticon-sentiment.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if '\t' in line:
                        emoticon, sentiment = line.split('\t', 1)
                        emoticon_dict[emoticon] = sentiment
        except FileNotFoundError:
            print("Warning: emoticon-sentiment.txt not found")
        return emoticon_dict
    
    def clean_text(self, text: str) -> str:
        """
        Membersihkan teks dari karakter yang tidak diinginkan
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'(\+62|62|0)[0-9]{8,13}', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep Indonesian punctuation
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        
        return text.strip()
    
    def normalize_text(self, text: str) -> str:
        """
        Normalisasi teks menggunakan dictionary normalisasi dan slang
        """
        words = text.split()
        normalized_words = []
        
        for word in words:
            # Check in normalization dictionary first
            if word in self.normalization_dict:
                normalized_word = self.normalization_dict[word]
            # Check in slang dictionary
            elif word in self.slang_dict:
                normalized_word = self.slang_dict[word]
            else:
                normalized_word = word
            
            normalized_words.append(normalized_word)
        
        return ' '.join(normalized_words)
    
    def remove_stopwords(self, text: str) -> str:
        """
        Menghapus stopwords dari teks
        """
        words = text.split()
        filtered_words = [word for word in words if word not in self.stopwords]
        return ' '.join(filtered_words)
    
    def preprocess_text(self, text: str) -> str:
        """
        Pipeline preprocessing lengkap
        """
        # Clean text
        text = self.clean_text(text)
        
        # Normalize text
        text = self.normalize_text(text)
        
        # Remove stopwords
        text = self.remove_stopwords(text)
        
        return text
    
    def preprocess_dataset(self, data: List[Dict]) -> List[Dict]:
        """
        Preprocess seluruh dataset
        """
        processed_data = []
        
        for item in data:
            processed_item = {
                'intent': item.get('intent', ''),
                'utterances': [],
                'answers': []
            }
            
            # Process utterances
            for utterance in item.get('utterances', []):
                processed_utterance = self.preprocess_text(utterance)
                if processed_utterance.strip():  # Only add non-empty utterances
                    processed_item['utterances'].append(processed_utterance)
            
            # Process answers
            for answer in item.get('answers', []):
                processed_answer = self.preprocess_text(answer)
                if processed_answer.strip():  # Only add non-empty answers
                    processed_item['answers'].append(processed_answer)
            
            # Only keep items with valid utterances and answers
            if processed_item['utterances'] and processed_item['answers']:
                processed_data.append(processed_item)
        
        return processed_data

def load_dataset_files() -> List[Dict]:
    """
    Load semua file dataset (JSON, TSV, TXT)
    """
    all_data = []
    
    # 1. Load JSON corpus files
    json_files = [
        'dataset/corpus/id/dialog.json',
        'dataset/corpus/id/agent.json', 
        'dataset/corpus/id/user.json',
        'dataset/corpus/id/motivasi.json',
        'dataset/corpus/id/None.json'
    ]
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data)
                print(f"Loaded {len(data)} items from {file_path}")
        except FileNotFoundError:
            print(f"Warning: {file_path} not found")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {file_path}")
    
    # 2. Load sentiment data (TSV)
    sentiment_files = [
        'dataset/sentiment/tsv/positive.tsv',
        'dataset/sentiment/tsv/negative.tsv'
    ]
    
    for file_path in sentiment_files:
        try:
            df = pd.read_csv(file_path, sep='\t', header=None, names=['text', 'label'])
            sentiment_data = []
            for _, row in df.iterrows():
                sentiment_data.append({
                    'intent': f'sentiment.{row["label"]}',
                    'utterances': [row['text']],
                    'answers': [f'Saya merasakan {row["label"]} tentang hal ini']
                })
            all_data.extend(sentiment_data)
            print(f"Loaded {len(sentiment_data)} sentiment items from {file_path}")
        except FileNotFoundError:
            print(f"Warning: {file_path} not found")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # 3. Load entities data
    try:
        with open('dataset/entities/json/example.json', 'r', encoding='utf-8') as f:
            entities_data = json.load(f)
            all_data.extend(entities_data)
            print(f"Loaded {len(entities_data)} entities from dataset/entities/json/example.json")
    except FileNotFoundError:
        print("Warning: dataset/entities/json/example.json not found")
    except json.JSONDecodeError:
        print("Error: Invalid JSON in entities file")
    
    return all_data

def create_training_pairs(data: List[Dict]) -> List[Tuple[str, str]]:
    """
    Membuat pasangan input-output untuk training chatbot yang lebih optimal
    """
    training_pairs = []
    
    for item in data:
        intent = item.get('intent', '')
        utterances = item.get('utterances', [])
        answers = item.get('answers', [])
        
        # Buat pasangan yang lebih spesifik berdasarkan intent
        for utterance in utterances:
            # Bersihkan utterance dari bracket notation
            clean_utterance = utterance.replace('[', '').replace(']', '').replace('|', ' ')
            clean_utterance = ' '.join(clean_utterance.split())  # Remove extra spaces
            
            # Tambahkan konteks untuk training yang lebih baik
            if answers:
                for answer in answers:
                    # Buat response yang lebih natural
                    if intent.startswith('agent.'):
                        # Untuk agent responses, buat lebih formal
                        enhanced_response = f"Baik, {answer.lower()}"
                    elif intent.startswith('user.'):
                        # Untuk user responses, buat lebih empati
                        enhanced_response = f"Saya mengerti, {answer.lower()}"
                    else:
                        enhanced_response = answer
                    
                    training_pairs.append((clean_utterance, enhanced_response))
                    
                    # Tambahkan variasi dengan greeting
                    if not clean_utterance.lower().startswith(('halo', 'hai', 'hi')):
                        training_pairs.append((f"halo {clean_utterance}", enhanced_response))
                        training_pairs.append((f"hai {clean_utterance}", enhanced_response))
    
    return training_pairs

if __name__ == "__main__":
    # Test preprocessing
    preprocessor = IndonesianTextPreprocessor(
        'dataset/normalization/normalization.txt',
        'dataset/normalization/stopword.txt'
    )
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset_files()
    print(f"Total items loaded: {len(dataset)}")
    
    # Preprocess dataset
    print("Preprocessing dataset...")
    processed_dataset = preprocessor.preprocess_dataset(dataset)
    print(f"Processed items: {len(processed_dataset)}")
    
    # Create training pairs
    training_pairs = create_training_pairs(processed_dataset)
    print(f"Training pairs created: {len(training_pairs)}")
    
    # Save processed data
    with open('processed_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(processed_dataset, f, ensure_ascii=False, indent=2)
    
    with open('training_pairs.json', 'w', encoding='utf-8') as f:
        json.dump(training_pairs, f, ensure_ascii=False, indent=2)
    
    print("Preprocessing completed!")
    print("Files saved: processed_dataset.json, training_pairs.json")
