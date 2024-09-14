from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

def grade_answer(llm_model, temperature=0):
    """
    Function to assess whether an answer is useful to resolve a user question.

    Args:
    - llm_model (str): The local language model to use (e.g., llama3).
    - temperature (float): The temperature setting for the LLM.

    Returns:
    - dict: A dictionary containing a binary 'yes' or 'no' score as 'score'.
    """
    
    # LLM setup
    llm = ChatOllama(model=llm_model, format="json", temperature=temperature)
    
    # Define the answer grading prompt
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
        You are a grader assessing whether an answer is useful to resolve a question. 
        Give a binary 'yes' or 'no' score to indicate whether the answer is useful to resolve a question. 
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the answer:\n ------- \n {generation} \n ------- \n
        Here is the question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "generation"]
    )
    
    # Create answer grader chain
    answer_grader = prompt | llm | JsonOutputParser()
    
    return answer_grader

