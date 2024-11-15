class OpenAIUsagePreparer:
    def __init__(self, prompt: str, response_metadata: dict):
        """
        Initialize with the prompt and response metadata from OpenAI.
        :param prompt: The prompt sent to the OpenAI API.
        :param response_metadata: The usage metadata from an OpenAI API response.
        """
        self.prompt = prompt

        # Access token usage from the nested dictionary
        token_usage = response_metadata.get('token_usage', {})
        self.input_tokens = token_usage.get('input_tokens')
        self.output_tokens = token_usage.get('output_tokens')
        self.total_tokens = token_usage.get('total_tokens')

        self.model_name = response_metadata.get('model_name')
        self.system_fingerprint = response_metadata.get('system_fingerprint')
        self.finish_reason = response_metadata.get('finish_reason')
        self.response_id = response_metadata.get('id')

    def prepare_for_logging(self):
        """
        Prepares the prompt and metadata for logging.
        :return: A dictionary ready to be logged.
        """
        if self.input_tokens is None or self.output_tokens is None or self.total_tokens is None:
            raise ValueError("Token usage data is missing or could not be accessed.")

        prepared_data = {
            "prompt": self.prompt,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "model_name": self.model_name,
            "system_fingerprint": self.system_fingerprint,
            "finish_reason": self.finish_reason,
            "response_id": self.response_id
        }
        return prepared_data
