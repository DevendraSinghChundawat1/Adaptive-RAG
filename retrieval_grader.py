from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

def grade_retrieved_document(llm_model, temperature=0):
    """
    Function to grade the relevance of a retrieved document to a user question using a binary 'yes' or 'no' score.

    Args:
    - llm_model (str): The local language model to use (e.g., llama3).
    - temperature (float): The temperature setting for the LLM.

    Returns:
    - dict: a rag chain
    """
    
    # LLM setup
    llm = ChatOllama(model=llm_model, format="json", temperature=temperature)
    
    # Define the grading prompt
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
        You are a grader assessing relevance of a retrieved document to a user question. 
        If the document contains keywords related to the user question, grade it as relevant. 
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation. 
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "document"]
    )
    
    # Create retrieval grader chain
    retrieval_grader = prompt | llm | JsonOutputParser()
    
    # Grade the document
    return retrieval_grader


