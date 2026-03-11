from haystack.dataclasses import ChatMessage
from haystack.components.builders import ChatPromptBuilder
from prompts.prompt import INSTRUCTION_PROMPT

def get_prompt_builder():
    # prompt = get_absa_prompt
    template = [ChatMessage.from_user(INSTRUCTION_PROMPT)]

    prompt_builder = ChatPromptBuilder(template=template, required_variables=["query", "documents"])
    
    return prompt_builder





    