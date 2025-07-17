import os
from dotenv import load_dotenv
import sys

# Import the custom exception class from your exception.py Canvas
from exception import CustomException
# Import the configured logger from your logger.py Canvas
from logger import logger

from llama_index.llms.gemini import Gemini
# IPython.display is not needed in a backend/API file
# from IPython.display import Markdown, display
import google.generativeai as genai

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Google Generative AI with your API key
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY not found in .env file. Model cannot be loaded.")
    # In a real API, you might raise an error or handle it differently
    # For this context, we'll just log and let the subsequent Gemini call potentially fail
genai.configure(api_key=GOOGLE_API_KEY)
logger.info("Google Generative AI configured in model_api.py.")

def load_model():
    """
    Loads a Gemini model for natural language processing.
    Uses 'gemini-2.0-flash' as the default model for better availability and performance
    in the context of this project.

    Returns:
    - Gemini: An instance of the Gemini class initialized with the specified model.
    """
    try:
        logger.info("Attempting to load Gemini LLM model (gemini-2.0-flash).")
        # Changed model to 'gemini-2.0-flash' which is generally more available
        model = Gemini(model='gemini-2.0-flash', api_key=GOOGLE_API_KEY)
        logger.info("Gemini LLM model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading Gemini LLM model: {e}")
        # Raise your custom exception with detailed error information
        raise CustomException(e, sys) from e
