from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

def grade_hallucination(llm_model, temperature=0):
    """
    Function to assess whether an answer is grounded in / supported by a set of facts.

    Args:
    - llm_model (str): The local language model to use (e.g., llama3).
    - temperature (float): The temperature setting for the LLM.

    Returns:
    - dict: A dictionary containing a binary 'yes' or 'no' score as 'score'.
    """
    
    # LLM setup
    llm = ChatOllama(model=llm_model, format="json", temperature=temperature)
    
    # Define the hallucination grading prompt
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
        You are a grader assessing whether an answer is grounded in / supported by a set of facts. 
        Give a binary 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. 
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation. 
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here are the facts:\n ------- \n {documents} \n ------- \n
        Here is the answer: {generation} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["documents", "generation"]
    )
    
    # Create hallucination grader chain
    hallucination_grader = prompt | llm | JsonOutputParser()
    
    return hallucination_grader
