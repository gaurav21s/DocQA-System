# Project Title: DocQA System

## Overview
The DocQA System is an AI-powered question-answering application designed to assist users in retrieving information from a specific company's or project's documentation. It leverages advanced language models to generate accurate and concise answers based on the provided context and keywords.

## Features
- **Web Crawling**: Automatically crawls documentation from specified URLs to gather relevant content.
- **Keyword Generation**: Generates keywords from user questions to enhance search relevance.
- **Contextual Answering**: Provides detailed answers based on the context retrieved from the documentation.
- **User-Friendly Interface**: Simple and intuitive web interface for users to interact with the system.

## Data Handling
The system utilizes a SQLite database to store crawled documentation. The `DataHandler` class manages the database operations, including creating tables, storing embeddings, and retrieving documents. Data is chunked into manageable pieces, and embeddings are generated using the SentenceTransformer model. This allows for efficient storage and retrieval of relevant information.

## Hybrid Search and Reranking
The retrieval process employs a hybrid search strategy that combines two techniques: BM25 and FAISS. 

1. **BM25**: This is a probabilistic model that ranks documents based on the term frequency and inverse document frequency. It is effective for keyword-based searches and provides an initial ranking of documents.

2. **FAISS**: This is a library for efficient similarity search and clustering of dense vectors. After encoding the documents into embeddings, FAISS is used to quickly retrieve the most similar documents based on the user's query.

After the initial retrieval, the results are reranked using a cross-encoder model. This model evaluates the relevance of the documents in the context of the user's question, ensuring that the most pertinent information is presented.

## Getting Started

### Prerequisites
- Python 3.7 or higher
- Flask
- Required Python packages (listed in `requirements.txt`)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/docqa-system.git
   cd docqa-system
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the root directory and add your `GROQ_API_KEY`:
   ```
   GROQ_API_KEY=your_api_key_here
   ```

### Running the Application
1. Start the Flask server:
   ```bash
   python main.py
   ```

2. Open your web browser and navigate to `http://127.0.0.1:5000`.

### Usage
- **Crawl Documentation**: Use the `/crawl` endpoint to start crawling documentation from a specified URL.
- **Process Data**: After crawling, use the `/process` endpoint to process and store the crawled data.
- **Ask Questions**: Use the `/ask` endpoint to submit questions and receive answers based on the crawled documentation.
- **Clear Database**: Use the `/clear` endpoint to reset the database and remove all stored data.

## Conclusion
The DocQA System integrates advanced AI techniques to provide a robust solution for retrieving and answering questions based on documentation. By combining web crawling, hybrid search, and contextual answering, it offers users a powerful tool for information retrieval.

## Video Overview
You can watch a brief usage of the DocQA System in the video below:

![DocQA System Overview](file:sample_video.mp4)



