import mysql.connector
import json
import pickle
import zlib
from datetime import datetime

class OpenAITokenUsageLogger:
    def __init__(self, db_config: dict, project_id: str = None, environment_id: str = None, user_id: str = None):
        # Default IDs for "unknown" project, environment, and user
        default_project_id = ""
        default_environment_id = ""
        default_user_id = ""

        # Assign provided IDs or fall back to default IDs
        self.project_id = project_id if project_id else default_project_id
        self.environment_id = environment_id if environment_id else default_environment_id
        self.user_id = user_id if user_id else default_user_id

        # Initialize MySQL connection
        self.db_connection = mysql.connector.connect(**db_config)
        self.cursor = self.db_connection.cursor()

    def compress_and_pickle(self, data: str) -> bytes:
        """
        Compress and serialize the given data using pickle and zlib.
        :param data: The prompt text to compress and serialize.
        :return: Compressed and serialized data as bytes.
        """
        serialized_data = pickle.dumps(data)  # Serialize the data using pickle
        compressed_data = zlib.compress(serialized_data)  # Compress the serialized data
        return compressed_data

    def log_usage(self, model: str, prompt: str, response_metadata: dict):
        """
        Log the usage data into the MySQL database.
        :param model: The OpenAI model used.
        :param prompt: The prompt sent to the model.
        :param response_metadata: The metadata from the OpenAI response.
        """
        timestamp = datetime.utcnow().isoformat()

        # Compress and serialize the prompt before storing it
        compressed_prompt = self.compress_and_pickle(prompt)

        # Remove or replace the 'prompt' field from the response_metadata if it exists
        if 'prompt' in response_metadata:
            del response_metadata['prompt']
        
        response_metadata_json = json.dumps(response_metadata)

        query = """
        INSERT INTO token_usage_logs 
        (id, project_id, environment_id, user_id, model, prompt, response_metadata, timestamp) 
        VALUES (UUID(), %s, %s, %s, %s, %s, %s, %s)
        """
        data = (
            self.project_id,
            self.environment_id,
            self.user_id,
            model,
            compressed_prompt,  # Store the compressed and serialized prompt
            response_metadata_json,
            timestamp
        )
        self.cursor.execute(query, data)
        self.db_connection.commit()

    def close(self):
        """
        Close the MySQL connection.
        """
        self.cursor.close()
        self.db_connection.close()
