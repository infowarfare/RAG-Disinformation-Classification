from haystack_integrations.components.generators.anthropic import AnthropicGenerator
from dotenv import load_dotenv
# Load API Keys
load_dotenv() 

def get_anthropic_generator():
    generator = AnthropicGenerator(model="claude-sonnet-4-6")
    return generator