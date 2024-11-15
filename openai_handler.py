import sys
from langchain_openai import ChatOpenAI

class OpenAIChatHandler:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.llm = ChatOpenAI(model=model, openai_api_key=api_key)
    def invoke_chat(self, prompt: str):
        response = self.llm.invoke(prompt)
        if response:
            # Combine additional metadata into the response_metadata
            response_metadata = {
                'token_usage': response.usage_metadata,
                'model_name': response.response_metadata.get('model_name'),
                'system_fingerprint': response.response_metadata.get('system_fingerprint'),
                'finish_reason': response.response_metadata.get('finish_reason'),
                'id': response.id
            }
            return response.content, response_metadata
        else:
            print("No response received. Exiting the program.")
            sys.exit(1)  # Exit the program with a non-zero status to indicate an error

