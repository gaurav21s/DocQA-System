from flask import Flask, render_template, request, jsonify
from utils.web_crawl import run_spider
from utils.data_new import DataHandler
from utils.retrieve_new import Retrieval
from utils.qna import LLMAssistant
from utils.logger import logger

app = Flask(__name__)

# Global variables to manage state
data_handler = None
retriever = None
assistant = LLMAssistant()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/crawl', methods=['POST'])
def crawl_documentation():
    global data_handler
    global start_url
    start_url = request.json.get('start_url')
    max_depth = int(request.json.get('max_depth', 3))

    if not start_url:
        return jsonify({'error': 'Please enter a URL before crawling.'}), 400

    if not start_url.startswith(('http://', 'https://')):
        return jsonify({'error': 'Please enter a valid URL starting with http:// or https://'}), 400
    
    data_handler = DataHandler()
    try:
        run_spider(start_url, max_depth=max_depth)
        logger.info("Crawling completed successfully!")
        return jsonify({'message': 'Crawling completed successfully!'})
    except Exception as e:
        logger.error(f"Error during crawling: {str(e)}")
        return jsonify({'error': f"An error occurred during crawling: {str(e)}"}), 500

@app.route('/process', methods=['POST'])
def process_data():
    global data_handler, retriever

    if not data_handler:
        return jsonify({'error': 'Please crawl data before processing.'}), 400

    try:
        data_handler.process_and_store('output.json')
        retriever = Retrieval(data_handler)
        logger.info("Data processed and stored in the database.")
        return jsonify({'message': 'Data processed successfully!'})
    except Exception as e:
        logger.error(f"Error during data processing: {str(e)}")
        return jsonify({'error': f"An error occurred during data processing: {str(e)}"}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    global data_handler, retriever
    global start_url
    question = request.json.get('question')

    if not question:
        return jsonify({'error': 'Please enter a question.'}), 400

    if not data_handler or not retriever:
        return jsonify({'error': 'Please crawl and process data before asking questions.'}), 400

    try:
        results, keywords = retriever.retrieve_and_rerank(question, assistant, start_url)
        context = " ".join(
            data_handler.get_doc_by_id(result['id'])['content'] 
            for result in results[:5] if data_handler.get_doc_by_id(result['id'])
        )
        context = context[:15000]  # Limit context

        answer = assistant.generate_answer(question, context, keywords)
        answer = str(answer).replace("assistant: ", "")
        
        documents = []
        for result in results[:5]:
            doc = data_handler.get_doc_by_id(result['id'])
            if doc:
                documents.append({
                    'title': doc['title'],
                    'url': doc['url'],
                    'content': doc['content'][:500],  # Truncate content
                    'id': result['id'],
                    'score': result['rerank_score']
                })

        return jsonify({'answer': answer, 'documents': documents})
    except Exception as e:
        logger.error(f"Error during question processing: {str(e)}")
        return jsonify({'error': f"An error occurred while processing your question: {str(e)}"}), 500

@app.route('/clear', methods=['POST'])
def clear_database():
    global data_handler, retriever

    if data_handler:
        try:
            data_handler.clear_collection()
            data_handler = None
            retriever = None
            logger.info("Database cleared successfully!")
            return jsonify({'message': 'Database cleared successfully!'})
        except Exception as e:
            logger.error(f"Error clearing the database: {str(e)}")
            return jsonify({'error': f"An error occurred while clearing the database: {str(e)}"}), 500
    else:
        return jsonify({'error': 'Data handler not initialized. Please crawl data first.'}), 400

if __name__ == '__main__':
    app.run(debug=True)