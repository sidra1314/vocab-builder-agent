# Import required packages
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_default_openai_client, set_tracing_disabled, AsyncOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get Gemini API key
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Setup Gemini client using OpenAI-style wrapper
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Set defaults
set_default_openai_client(external_client)
set_tracing_disabled(True)

# Initialize the model with Gemini config
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",  
    openai_client=external_client
)

# Function to write markdown file
def write_markdown(word, response):
    file_path = "output.md"
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"#  Vocabulary Builder\n\n")
        f.write(f"**Word:** `{word}`\n\n")
        f.write(f"```\n{response}\n```")
    
    print(f"\n Result saved to {file_path}\n")

# Function to run vocab wizard
def vocab_wizard_agent():
    # Get word from user
    word = input(" Enter an English word: ")

    # Define agent
    agent = Agent(
        name="vocab_wizard",
        instructions=(
            "You are a Vocabulary Builder Wizard. "
            "For every input word, return the following:\n"
            "- Word\n"
            "- Meaning\n"
            "- Part of Speech\n"
            "- Example Sentence\n\n"
            "Reply only in that format."
        ),
        model=model
    )

    # Run agent
    result = Runner.run_sync(agent, word)
    
    # Print result in terminal
    print("\n Agent Response:\n")
    print(result.final_output)
    
    # Save to markdown
    write_markdown(word, result.final_output)

# Entry point
if __name__ == "__main__":
    vocab_wizard_agent()



