# main.py
from openai_handler import OpenAIChatHandler
from usage_logger import OpenAITokenUsageLogger
from openai_usage_preparer import OpenAIUsagePreparer
import os

db_config = {
    'user': os.getenv('open_ai_user_updater'),
    'password': os.getenv('open_ai_updater_password'),
    'host': os.getenv('host_name'),
    'database': 'open_ai'
}

api_key = os.getenv('open_ai_api_key')
model = 'gpt-3.5-turbo'  # Example model, you can specify any model

# Set project and environment variables
project_id = ""
environment_id = ""
user_id = "" 

# Print statement to verify execution
print("Starting main.py execution...")

# Initialize the OpenAI handler and MySQL logger
handler = OpenAIChatHandler(api_key=api_key, model=model)
logger = OpenAITokenUsageLogger(db_config=db_config, project_id=project_id, environment_id=environment_id, user_id=user_id)

# Make a request to the OpenAI API and log the usage
prompt = "Hi there"
response_text, usage_metadata = handler.invoke_chat(prompt)

print(response_text)

# Initialize the preparer
preparer = OpenAIUsagePreparer(prompt, usage_metadata)

# Prepare data for logging
prepared_data = preparer.prepare_for_logging()

# Now, you can pass prepared_data to your logger
logger = OpenAITokenUsageLogger(db_config=db_config, project_id=project_id, environment_id=environment_id, user_id=user_id)
logger.log_usage(model=prepared_data['model_name'], prompt=prepared_data['prompt'], response_metadata=prepared_data)
logger.close()