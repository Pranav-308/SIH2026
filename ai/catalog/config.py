import os
from typing import Optional
from dotenv import load_dotenv
from exceptions import MissingAPIKeyError

# Load environment variables from .env file in current or parent directories
load_dotenv()

class Config:
    """Application configuration management."""
    
    @staticmethod
    def get_api_key(strict: bool = True) -> Optional[str]:
        """
        Retrieves GEMINI_API_KEY from environment variables.
        
        :param strict: If True, raises MissingAPIKeyError when API key is absent or empty.
        :return: API key string or None (if strict=False).
        """
        key = os.getenv("GEMINI_API_KEY")
        if not key or not key.strip():
            if strict:
                raise MissingAPIKeyError("GEMINI_API_KEY environment variable is not set. Please set it in .env or environment.")
            return None
        return key.strip()

    @staticmethod
    def get_model_name() -> str:
        """Retrieves target Gemini model name."""
        val = os.getenv("GEMINI_MODEL")
        return (val if val else "gemini-3.6-flash").strip()

    @staticmethod
    def get_host() -> str:
        val = os.getenv("HOST")
        return (val if val else "0.0.0.0").strip()

    @staticmethod
    def get_port() -> int:
        try:
            val = os.getenv("PORT")
            return int(val) if val else 8000
        except ValueError:
            return 8000
