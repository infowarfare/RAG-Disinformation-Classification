from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator
from dotenv import load_dotenv

# Load API Keys
load_dotenv() 

def get_gemini_generator():
    chat_generator = GoogleGenAIChatGenerator(model="gemini-3-flash-preview")
    return chat_generator

