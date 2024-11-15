import os
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from usage_logger import OpenAITokenUsageLogger
from llama_index_usage_preparer import LLaMaIndexUsagePreparer

# Configuration for MySQL database
db_config = {
    'user': os.getenv('open_ai_user_updater'),
    'password': os.getenv('open_ai_updater_password'),
    'host': os.getenv('host_name'),
    'database': 'open_ai'
}

# API Key and model configuration
api_key = os.getenv('open_ai_api_key')
model = 'gpt-3.5-turbo'  # Example model, you can specify any model

# Set project and environment variables
project_id = ""
environment_id = ""
user_id = ""

# Print statement to verify execution
print("Starting main.py execution...")

# Initialize the LLaMa Index handler and MySQL logger
llm = OpenAI(model=model, api_key=api_key)
logger = OpenAITokenUsageLogger(db_config=db_config, project_id=project_id, environment_id=environment_id, user_id=user_id)

# Create the prompt
prompt = "Are you tired?"
messages = [
    ChatMessage(role="system", content="You are a helpful assistant."),
    ChatMessage(role="user", content=prompt)
]

# Make a request to the LLaMa Index API and log the usage
try:
    resp = llm.chat(messages)
except Exception as e:
    print(f"Error: {str(e)}")
    resp = None

if resp:
    print(resp.message.content)
    print(resp.raw.usage)

    # Extract metadata and log it
    preparer = LLaMaIndexUsagePreparer(prompt, resp)

    prepared_data = preparer.prepare_for_logging()

    # Log the data to the MySQL database
    logger.log_usage(model=prepared_data['model_name'], prompt=prepared_data['prompt'], response_metadata=prepared_data)

# Close the logger connection when done
logger.close()

# Print statement to verify execution
print("Successfully finished!")
