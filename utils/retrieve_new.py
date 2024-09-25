import os
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, DPRQuestionEncoder
from rank_bm25 import BM25Okapi
import torch
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np
import warnings
from utils.data import DataHandler
from utils.logger import logger
import faiss
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from sklearn.decomposition import TruncatedSVD
import string

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Retrieval:
    def __init__(self, data_handler):
        logger.info("Initializing Retrieval class")
        
        self.all_docs = data_handler.get_all_docs()
        self.doc_embeddings = np.array([doc['embedding'] for doc in self.all_docs])
        self.doc_contents = [doc['content'] for doc in self.all_docs]
        self.doc_ids = [doc['id'] for doc in self.all_docs]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
            self.question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").cpu()
        self.cross_encoder = SentenceTransformer('all-MiniLM-L6-v2')

        self.stemmer = PorterStemmer()
        self.bm25 = self._initialize_bm25()

        # Dimensionality reduction
        self.svd = TruncatedSVD(n_components=256)
        self.reduced_embeddings = self.svd.fit_transform(self.doc_embeddings)
        
        # FAISS index for fast similarity search
        d = self.reduced_embeddings.shape[1]  # Get the number of dimensions
        self.index = faiss.IndexFlatIP(d)
        self.index.add(self.reduced_embeddings.astype('float32'))

    def _preprocess_text(self, text: str) -> List[str]:
        try:
            text = text.lower().translate(str.maketrans('', '', string.punctuation))
            tokens = word_tokenize(text)
            return [self.stemmer.stem(token) for token in tokens]
        except LookupError as e:
            logger.error(f"NLTK resource not found: {e}")
            # Fallback to a simple tokenization
            return text.split()

    def _initialize_bm25(self) -> BM25Okapi:
        tokenized_corpus = [self._preprocess_text(doc) for doc in self.doc_contents]
        return BM25Okapi(tokenized_corpus)

    def query_expansion(self, query: str) -> List[str]:
        expanded_queries = [query]
        for word in query.split():
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    expanded_queries.append(query.replace(word, lemma.name()))
        return list(set(expanded_queries))

    @lru_cache(maxsize=1000)
    def bm25_retrieval(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
        tokenized_query = self._preprocess_text(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_n = sorted(zip(self.doc_ids, doc_scores), key=lambda x: x[1], reverse=True)[:top_k]
        return top_n

    def encode_question(self, question: str) -> np.ndarray:
        inputs = self.tokenizer(question, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            outputs = self.question_encoder(**inputs)
            question_embedding = outputs.pooler_output.cpu().numpy()
        return question_embedding.flatten()

    def faiss_retrieval(self, query: str, top_k: int = 100) -> List[Dict]:
        question_embedding = self.encode_question(query)
        reduced_query = self.svd.transform(question_embedding.reshape(1, -1)).astype('float32')
        D, I = self.index.search(reduced_query, top_k)
        
        return [{'id': self.doc_ids[i], 'score': float(D[0][j]), 
                 'url': self.all_docs[i]['url'], 'title': self.all_docs[i]['title']} 
                for j, i in enumerate(I[0])]

    def hybrid_retrieval(self, query: str, top_k: int = 100) -> List[Dict]:
        logger.info("Retrieving results from BM25 and FAISS")
        bm25_results = self.bm25_retrieval(query, top_k=top_k)
        faiss_results = self.faiss_retrieval(query, top_k=top_k)
        
        combined_results = {}
        max_bm25_score = max((score for _, score in bm25_results), default=1)
        max_faiss_score = max((hit['score'] for hit in faiss_results), default=1)
        
        for i, score in bm25_results:
            combined_results[i] = score / max_bm25_score if max_bm25_score != 0 else 0
        
        for hit in faiss_results:
            doc_id = hit['id']
            if doc_id in combined_results:
                combined_results[doc_id] += hit['score'] / max_faiss_score if max_faiss_score != 0 else 0
            else:
                combined_results[doc_id] = hit['score'] / max_faiss_score if max_faiss_score != 0 else 0
        
        sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"id": doc_id, "score": score} for doc_id, score in sorted_results]

    def parallel_encode(self, texts: List[str]) -> List[np.ndarray]:
        with ThreadPoolExecutor() as executor:
            return list(executor.map(self.cross_encoder.encode, texts))

    def rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        logger.info("Reranking the results")
        pairs = [(query, self.all_docs[self.doc_ids.index(result['id'])]['content']) for result in results]
        
        # Batch encoding
        query_embeddings = self.parallel_encode([pair[0] for pair in pairs])
        doc_embeddings = self.parallel_encode([pair[1] for pair in pairs])
        
        similarity_scores = util.pytorch_cos_sim(query_embeddings, doc_embeddings)
        for i, score in enumerate(similarity_scores[0]):
            results[i]['rerank_score'] = float(score)
        reranked_results = sorted(results, key=lambda x: x.get('rerank_score', 0), reverse=True)
        return reranked_results

    def retrieve_and_rerank(self, query: str, assistant, url, top_k: int = 50) -> Tuple[List[Dict], List[str]]:
        keywords = assistant.generate_keyword(query, url)
        logger.info(f"Generated keywords: {keywords}")
        expanded_queries = set(self.query_expansion(query) + keywords)
        
        # Sequential processing of expanded queries
        all_results = []
        for q in expanded_queries:
            all_results.extend(self.hybrid_retrieval(q, top_k))
        
        unique_results = list({r['id']: r for r in all_results}.values())
        top_results = sorted(unique_results, key=lambda x: x['score'], reverse=True)[:top_k]
        final_results = self.rerank(query, top_results)
        logger.info(f"Retrieved and reranked {len(final_results)} results")
        return final_results, keywords

if __name__ == "__main__":
    # Example usage
    assistant = ""
    data_handler = DataHandler("company_docs")
    retriever = Retrieval(data_handler)
    query = "How to optimize CUDA kernel performance?"
    results, keywords = retriever.retrieve_and_rerank(query, assistant)
    for i, result in enumerate(results[:10], 1):
        print(f"{i}. Document ID: {result['id']}, Score: {result['rerank_score']:.4f}")
    print(f"Generated keywords: {keywords}")