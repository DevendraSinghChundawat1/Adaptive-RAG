from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate


def route_question(llm_model, temperature=0):
    """
    Function to route a question to either a vectorstore or web search.

    Args:
    - llm_model (str): The local language model to use (e.g., llama3).
    - temperature (float): The temperature setting for the LLM.

    Returns:
    - dict: A dictionary containing the chosen datasource as 'datasource'.
    """
    
    # LLM setup
    llm = ChatOllama(model=llm_model, format="json", temperature=temperature)
    
    # Define the question routing prompt
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
        You are an expert at routing a user question to a vectorstore or web search. 
        Use the vectorstore for questions on LLM agents, prompt engineering, and adversarial attacks. 
        You do not need to be stringent with the keywords in the question related to these topics. 
        Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question. 
        Return the a JSON with a single key 'datasource' and no preamble or explanation.
        Question to route: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"]
    )
    
    # Create question router chain
    question_router = prompt | llm | JsonOutputParser()
    
    return question_router
