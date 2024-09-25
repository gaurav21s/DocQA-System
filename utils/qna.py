from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage
import os
from utils.logger import logger
from dotenv import load_dotenv
from typing import List

load_dotenv()

class LLMAssistant:
    def __init__(self):
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        self.llm = Groq(model="llama-3.1-70b-versatile", api_key=GROQ_API_KEY, temperature=0.0)
        logger.info("Initialized Groq AI")

    def generate_answer(self, question: str, context: str, keywords: List[str]) -> str:
        logger.info(f"Generating answers from LLM")
        system_prompt = (
            f"""
            "You are an AI assistant specialized in answering questions about a specific company's or project's documentation."
            "Understand the question and what it meant and use context to provide answers."
            "Use the provided context to answer the user's question accurately, concisely, and in detail."
            "You can understand the question better using these keywords {keywords}."
            "Your response should be concise yet thorough. If the context does not contain the answer, state: 'I don't have enough information to answer this question.'"
            "Do not speculate, do not suggest ideas on your own or do not rely on external knowledge."
            "Only When context is available, be creative in your response, enhancing clarity and quality, while strictly adhering to the provided information."
            "Correct any grammatical or spacing errors in the answer."
            """
        )
        user_message = f"Context: {context}\n\nQuestion: {question}"
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_message),
        ]
        response = self.llm.chat(messages)
        return str(response)

    def generate_keyword(self, question: str, url: str) -> List[str]:
        """
        Generate keywords from the given question.
        Args:
            question (str): The input question.
        Returns:
            List[str]: A list of generated keywords.
        """
        logger.info("Generating keywords for question")
        system_prompt = f"""
    You are an AI assistant specialized in generating keywords for the specific company's or project's documentation based on the question.
    The documentation url is {url}. Use this to generate keywords.
    Your task is to convert the given question into 3 keywords that can be used to find relevant text or content from documentation.
    Provide only the keywords, one per line, without any additional text or explanation.
    Give your output strictly like below:
    Keyword1
    Keyword2
    Keyword3
    """
        user_message = f"Question: {question}\n\nGenerate 3 keywords:"
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_message),
        ]
        response = self.llm.chat(messages)
        response = str(response).replace("assistant:","")
        # print('in the function:',response)
        keywords = str(response).strip().split('\n')
        # logger.info(f"Generated keywords: {keywords}")
        return keywords[:3]

# if __name__ == "__main__":
#     assistant = LLMAssistant()
#     question = "What is CUDA?"
#     context = "CUDA is a parallel computing platform and programming model developed by NVIDIA for general computing on graphical processing units (GPUs). With CUDA, developers can dramatically speed up computing applications by harnessing the power of GPUs."
#     answer = assistant.generate_answer(question, context)
#     print(f"Question: {question}")
#     print(f"Answer: {answer}")

    # assistant = LLMAssistant()
    # questions = [
    #     "What is CUDA?",
    #     "How does CUDA handle memory management?",
    #     "What are the benefits of using CUDA?",
    # ]
    # for question in questions:
    #     keywords = assistant.generate_keyword(question)
    #     print(f"Question: {question}")
    #     print(f"Keywords: {keywords}")
    #     print("------------------------------------------------")
