import sys
# Import the custom exception class from your exception.py Canvas
from exception import CustomException
# Import the configured logger from your logger.py Canvas
from logger import logger

from llama_index.core import VectorStoreIndex
# ServiceContext is deprecated with global Settings, so we remove it
# from llama_index.core import ServiceContext
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.settings import Settings # Import Settings for global configuration

# Assuming these modules exist in your project structure
# from QAWithPDF.data_ingestion import load_data
# from QAWithPDF.model_api import load_model

def download_gemini_embedding(model, document):
    """
    Downloads and initializes a Gemini Embedding model for vector embeddings,
    builds a VectorStoreIndex from the provided document, and creates a query engine.

    Args:
        model: The LLM model (e.g., Gemini) to be used by the query engine.
        document: The document(s) (e.g., from SimpleDirectoryReader) to index.

    Returns:
    - query_engine: An LlamaIndex QueryEngine for efficient similarity queries.
    """
    try:
        logger.info("Initializing Gemini Embedding model.")
        # Ensure the embedding model is consistent with what's set globally in app.py
        # Using 'models/text-embedding-004' as agreed upon for better availability
        gemini_embed_model = GeminiEmbedding(model_name="models/text-embedding-004")

        # Instead of ServiceContext, we rely on global Settings configured in app.py
        # Ensure Settings.llm and Settings.embed_model are set before calling this.
        # If you were to use ServiceContext here, it would override global settings
        # for this specific index, but for simplicity and consistency, we'll use global.
        Settings.embed_model = gemini_embed_model # Ensure this function uses the correct embed model

        logger.info("Building VectorStoreIndex from documents.")
        index = VectorStoreIndex.from_documents(document) # ServiceContext is not needed if Settings are global
        
        # Persisting the index is usually done outside the embedding function
        # if you want to load it later without re-indexing every time.
        # For this Streamlit app, we are re-creating the index on each PDF upload
        # and caching it with @st.cache_resource, so explicit persist() here might
        # not be strictly necessary unless you want to save it to disk for other uses.
        # index.storage_context.persist() # Uncomment if you need to save to disk

        logger.info("Creating query engine from the index.")
        query_engine = index.as_query_engine()
        return query_engine
    except Exception as e:
        # Raise your custom exception with detailed error information
        logger.error(f"Error in download_gemini_embedding function: {e}")
        raise CustomException(e, sys) from e

# Example usage (for testing this module independently if needed)
if __name__ == "__main__":
    # This block would typically be used for unit testing or demonstrating
    # the function's behavior in isolation.
    # For a full run, refer to the Streamlit app (StreamlitApp.py or app.py).

    # You would need to mock or provide actual 'model' and 'document' objects
    # For instance:
    # from llama_index.llms.gemini import Gemini
    # from llama_index.core import SimpleDirectoryReader
    # from dotenv import load_dotenv
    # import os

    # load_dotenv()
    # GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    # if not GOOGLE_API_KEY:
    #     print("GOOGLE_API_KEY not found. Cannot run example.")
    #     sys.exit(1)

    # genai.configure(api_key=GOOGLE_API_KEY)
    # Settings.llm = Gemini(api_key=GOOGLE_API_KEY, model="gemini-2.0-flash")
    # Settings.embed_model = GeminiEmbedding(api_key=GOOGLE_API_KEY, model_name="models/text-embedding-004")
    # Settings.chunk_size = 800
    # Settings.chunk_overlap = 20

    # try:
    #     # Create a dummy document for testing
    #     # Ensure you have a 'Data' directory with 'sample.txt' for this to work
    #     # documents = SimpleDirectoryReader("../Data").load_data()
    #     # if not documents:
    #     #     print("No documents found in ../Data/. Cannot run example.")
    #     #     sys.exit(1)

    #     # # Call the function
    #     # query_engine = download_gemini_embedding(Settings.llm, documents)
    #     # print("Query engine created successfully in example.")

    #     # # You can then try a query
    #     # response = query_engine.query("What is machine learning?")
    #     # print(f"Example Query Response: {response.response}")

    #     print("This module is designed to be imported by StreamlitApp.py or similar.")
    #     print("Example usage commented out as it requires external files and API key.")

    # except CustomException as ce:
    #     print(f"Caught custom exception in example: {ce}")
    # except Exception as e:
    #     print(f"Caught unexpected exception in example: {e}")
    pass # Keep pass if no example code is active
