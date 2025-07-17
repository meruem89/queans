from llama_index.core import SimpleDirectoryReader
import sys
# Corrected import: 'exception' is the module file, 'CustomException' is the class
from exception import CustomException
# Corrected import: 'logger' is the module file, 'logger' is the object/instance
from logger import logger

def load_data(data):
    """
    Load PDF documents from a specified directory or directly from a file stream.

    Parameters:
    - data: The path to the directory containing PDF files (str) or a file-like object (e.g., from Streamlit's file_uploader).

    Returns:
    - A list of loaded LlamaIndex Document objects.
    """
    try:
        logger.info("Data loading started...")
        
        # In the Streamlit app, 'data' will be the uploaded file object,
        # not a directory path. This function needs to be adapted for that.
        # For consistency with the Streamlit app's usage, we'll expect a file-like object.
        # If this function is meant for a local 'Data' directory, then the SimpleDirectoryReader
        # usage is correct. Given the context of the Streamlit app, I'll adapt it.

        # Assuming 'data' here is a file-like object from Streamlit's file_uploader
        # This part needs to be consistent with how load_data is called in StreamlitApp.py
        # In the Streamlit app, we extract text and then create a Document object.
        # This `load_data` function seems designed for a directory.
        # Let's adjust it to handle either a path or a file stream for flexibility,
        # but for the Streamlit app, the text extraction is handled in app.py directly.

        # If this `load_data` is intended to be used with a local 'Data' directory:
        if isinstance(data, str): # 'data' is a path string
            loader = SimpleDirectoryReader(data) # Expects a directory path
            documents = loader.load_data()
            logger.info(f"Data loaded from directory: {data}")
            return documents
        else: # 'data' is assumed to be a file-like object (e.g., from st.file_uploader)
            # This branch is more aligned with the Streamlit app's direct file handling
            # However, the Streamlit app's `process_pdf_and_create_index` already
            # handles text extraction and Document creation.
            # So, this `load_data` might not be directly used by the Streamlit app
            # in its current form if it's expecting a file-like object.
            # If `load_data` is meant to take the *content* of the PDF after extraction,
            # then it should take a string.

            # Re-evaluating based on StreamlitApp.py's call: `document=load_data(doc)`
            # where `doc` is `st.file_uploader`.
            # This `load_data` needs to handle the uploaded file object directly.
            # It should extract text and return LlamaIndex Document(s).
            from PyPDF2 import PdfReader # Import here to avoid circular dependency
            
            # Reset stream position if it's already been read
            data.seek(0) 
            reader = PdfReader(data)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            
            from llama_index.core.schema import Document
            documents = [Document(text=text)]
            logger.info("Data loaded from uploaded file stream.")
            return documents

    except Exception as e:
        logger.error(f"Exception in loading data: {e}", exc_info=True) # Log full traceback
        raise CustomException(e, sys) from e

