from pydantic import BaseModel
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from dotenv import load_dotenv

# Load API Keys
load_dotenv() 

class DisnformationClassification(BaseModel):
    disinformation_class: int

def get_openai_generator():
    generator = OpenAIChatGenerator(
        model="gpt-5.4-2026-03-05",
        generation_kwargs={"response_format": DisnformationClassification}
    )

    return generator