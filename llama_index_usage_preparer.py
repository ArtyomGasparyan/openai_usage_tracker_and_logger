from llama_index.core.base.llms.types import ChatResponse


class LLaMaIndexUsagePreparer:
    def __init__(self, prompt: str, response: ChatResponse):
        """
        Initialize with the prompt and response metadata from LLaMa Index.
        :param prompt: The prompt sent to the LLaMa Index API.
        :param response: The chat response object returned by LLaMa Index.
        """
        self.prompt = prompt
        self.response = response

    def prepare_for_logging(self):
        """
        Prepares the prompt and metadata for logging.
        :return: A dictionary ready to be logged.
        """
        prepared_data = {
            "prompt": self.prompt,
            "input_tokens": getattr(self.response.raw.usage, 'prompt_tokens', None),
            "output_tokens": getattr(self.response.raw.usage, 'completion_tokens', None),
            "total_tokens": getattr(self.response.raw.usage, 'total_tokens', None),
            "model_name": getattr(self.response.raw, 'model', None),
            "system_fingerprint": getattr(self.response.raw, 'system_fingerprint', None),
            "finish_reason": getattr(self.response.raw.choices[0], 'finish_reason', None),
            "response_id": getattr(self.response.raw, 'id', None)
        }
        if (prepared_data["input_tokens"] is None or
                prepared_data["output_tokens"] is None or
                prepared_data["total_tokens"] is None):
            raise ValueError("Token usage data is missing or could not be accessed.")
        return prepared_data
