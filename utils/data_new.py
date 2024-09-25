import sqlite3
import json
import os
import re
import numpy as np
from typing import List, Dict, Tuple
from collections import deque
from contextlib import contextmanager
from utils.logger import logger

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sentence_transformers import SentenceTransformer

# Ensure necessary NLTK data is downloaded
# nltk.download('punkt', quiet=True)
# nltk.download('stopwords', quiet=True)
# nltk.download('wordnet', quiet=True)

class DataHandler:
    def __init__(self, db_path: str = "embeddings.db", collection_name: str = "company_docs"):
        self.collection_name = collection_name
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.db_path = db_path
        self.create_table_if_not_exists()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    @contextmanager
    def db_connection(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def create_table_if_not_exists(self):
        with self.db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.collection_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT,
                title TEXT,
                content TEXT,
                embedding BLOB
            )
            """)
            conn.commit()
        logger.info(f"Table '{self.collection_name}' is ready.")

    def load_data(self, file_path: str) -> List[Dict]:
        with open(file_path, 'r') as f:
            return json.load(f)

    def preprocess_text(self, text: str) -> str:
        if text:
            text = text.lower()
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
            return ' '.join(tokens)
        else:
            return ''
            
    def advanced_chunking(self, documents: List[Dict], chunk_size: int = 256, overlap: int = 32) -> List[Tuple[str, str, str, str]]:
        logger.info("Chunking and refining the data")
        chunks = []
        for doc in documents:
            chunks.extend(self.chunk_single_document(doc, chunk_size, overlap))

        vectorizer = TfidfVectorizer(max_features=1000)
        tfidf = vectorizer.fit_transform([c[2] for c in chunks])  # Use content for TF-IDF
        lda = LatentDirichletAllocation(n_components=10, random_state=42)
        topic_distribution = lda.fit_transform(tfidf)

        refined_chunks = []
        for i, chunk in enumerate(chunks):
            main_topic = np.argmax(topic_distribution[i])
            if topic_distribution[i][main_topic] > 0.3:
                refined_chunks.append(chunk)

        return refined_chunks

    def chunk_single_document(self, doc: Dict, chunk_size: int, overlap: int) -> List[Tuple[str, str, str, str]]:
        chunks = []
        title = self.preprocess_text(doc['title'])
        content = self.preprocess_text(doc['content'])
        sentences = nltk.sent_tokenize(content)
        window = deque(maxlen=chunk_size)
        
        for i, sentence in enumerate(sentences):
            window.append(sentence)
            
            if len(window) == chunk_size:
                chunk_content = ' '.join(window)
                chunk_with_title = f"{title} {chunk_content}"
                
                if chunks and i >= chunk_size:
                    prev_chunk = chunks[-1][2]  # Get the previous chunk's content
                    similarity = self.model.encode([chunk_with_title, prev_chunk]).mean()
                    
                    if similarity > 0.8:
                        chunks[-1] = (doc['url'], doc['title'], chunks[-1][2] + ' ' + sentence, f"{title} {chunks[-1][2]} {sentence}")
                    else:
                        chunks.append((doc['url'], doc['title'], chunk_content, chunk_with_title))
                else:
                    chunks.append((doc['url'], doc['title'], chunk_content, chunk_with_title))
                
                # Slide the window
                for _ in range(chunk_size - overlap):
                    if sentences:
                        sentences.pop(0)
                    else:
                        break

        # Handle any remaining sentences
        if sentences:
            last_chunk_content = ' '.join(sentences)
            last_chunk_with_title = f"{title} {last_chunk_content}"
            chunks.append((doc['url'], doc['title'], last_chunk_content, last_chunk_with_title))

        return chunks

    def create_embeddings(self, chunks: List[Tuple[str, str, str, str]]) -> np.ndarray:
        return self.model.encode([c[3] for c in chunks])  # Use the title+content for embedding

    def store_in_db(self, embeddings: np.ndarray, chunks: List[Tuple[str, str, str, str]]):
        logger.info("Storing data in SQLite")
        with self.db_connection() as conn:
            cursor = conn.cursor()
            records = [
                (chunk[0], chunk[1], chunk[2], sqlite3.Binary(embedding.tobytes()))
                for embedding, chunk in zip(embeddings, chunks)
            ]
            cursor.executemany(f"""
            INSERT INTO {self.collection_name} (url, title, content, embedding) 
            VALUES (?, ?, ?, ?)""", records)
            conn.commit()

    def process_and_store(self, file_path: str):
        with self.db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {self.collection_name}")
            count = cursor.fetchone()[0]

        if count > 0:
            logger.info(f"Database already contains {count} records. Skipping chunking and preprocessing.")
            return

        documents = self.load_data(file_path)
        chunks = self.advanced_chunking(documents)
        embeddings = self.create_embeddings(chunks)
        self.store_in_db(embeddings, chunks)
        logger.info(f"Stored {len(chunks)} chunks in the SQLite database.")

    def get_all_docs(self):
        with self.db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT id, url, title, content, embedding FROM {self.collection_name}")
            rows = cursor.fetchall()
        
        return [
            {
                'id': row[0],
                'url': row[1],
                'title': row[2],
                'content': row[3],
                'embedding': np.frombuffer(row[4], dtype=np.float32)
            }
            for row in rows
        ]

    def get_doc_by_id(self, doc_id: int) -> Dict:
        with self.db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT id, url, title, content FROM {self.collection_name} WHERE id = ?", (doc_id,))
            row = cursor.fetchone()
        
        if row:
            return {
                'id': row[0],
                'url': row[1],
                'title': row[2],
                'content': row[3]
            }
        return None

    def clear_collection(self):
        logger.info("Clearing the collection")
        with self.db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {self.collection_name}")
            conn.commit()

        file_path = "output.json"
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"File {file_path} has been deleted.")
        else:
            logger.info(f"File {file_path} does not exist.")

if __name__ == "__main__":
    handler = DataHandler()
    handler.process_and_store('/Users/Admin/Desktop/Project/stepsai/final_app/output.json')