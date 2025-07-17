import streamlit as st
import os
import base64
import io
import json
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.schema import Document # Import Document for creating LlamaIndex Document objects
import google.generativeai as genai
from PyPDF2 import PdfReader # Used to extract text from PDF

# Load environment variables
load_dotenv()

# Configure Google Generative AI with your API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found in .env file. Please set it.")
    st.stop() # Stop the app if API key is missing

genai.configure(api_key=GOOGLE_API_KEY)

# --- Initialize Gemini Models for LlamaIndex ---
# Use st.cache_resource to ensure models are loaded only once
@st.cache_resource
def initialize_models():
    """Initializes and caches the Gemini LLM and Embedding models."""
    try:
        # Changed model to 'gemini-2.0-flash' which is generally more available
        gemini_llm = Gemini(api_key=GOOGLE_API_KEY, model="gemini-2.0-flash")
        # Changed embedding model to 'text-embedding-004' for better compatibility
        gemini_embed_model = GeminiEmbedding(api_key=GOOGLE_API_KEY, model_name="models/text-embedding-004")

        # Configure global settings for LlamaIndex
        Settings.llm = gemini_llm
        Settings.embed_model = gemini_embed_model
        Settings.chunk_size = 800
        Settings.chunk_overlap = 20
        return True
    except Exception as e:
        st.error(f"Error initializing Gemini models: {e}. Please check your API key and model availability.")
        return False

# --- Functions for PDF Processing, MCQ Generation, and Chatbot Querying ---

def extract_text_from_pdf_stream(pdf_file_stream) -> str:
    """
    Extracts text from an uploaded PDF file stream.
    """
    try:
        reader = PdfReader(pdf_file_stream)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or "" # Handle pages with no extractable text
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

@st.cache_resource
def process_pdf_and_create_index(uploaded_file):
    """
    Processes an uploaded PDF file, extracts text, and creates a LlamaIndex VectorStoreIndex.
    Caches the index to avoid re-processing on every rerun.
    """
    st.info("Processing PDF and building knowledge base... This may take a moment.")
    pdf_text = extract_text_from_pdf_stream(uploaded_file)

    if not pdf_text:
        st.error("No text extracted from PDF. Cannot create knowledge base.")
        return None

    # LlamaIndex expects a list of Document objects
    # For simplicity, we create a single document from the entire extracted text.
    # For very large PDFs, you might want to split this text into multiple documents.
    documents = [Document(text=pdf_text)]

    try:
        index = VectorStoreIndex.from_documents(documents)
        st.success("Knowledge base built successfully!")
        return index
    except Exception as e:
        st.error(f"Error building knowledge base: {e}")
        return None

def generate_mcqs_from_pdf(index: VectorStoreIndex, num_questions: int = 5, difficulty: str = "medium"):
    """
    Generates multiple-choice questions from the content indexed from the PDF.
    The response format is requested as JSON.
    """
    if index is None:
        st.error("PDF knowledge base not available. Please process a PDF first.")
        return []

    prompt = f"""
    Generate {num_questions} multiple-choice questions based on the provided document content.
    Each question should have 4 options (A, B, C, D) and clearly indicate the correct answer.
    The difficulty level should be {difficulty}.

    Provide the output as a JSON array of objects. Each object should have the following structure:
    {{
        "question": "The question text",
        "options": ["Option A", "Option B", "Option C", "Option D"],
        "correctAnswer": "The correct option (e.g., 'Option B')"
    }}
    Ensure the JSON is valid and can be directly parsed.
    """

    st.info(f"Generating {num_questions} MCQs with difficulty '{difficulty}'...")
    try:
        query_engine = index.as_query_engine()
        response = query_engine.query(prompt)

        # The model should return a JSON string. We need to parse it.
        # Sometimes, the model might include extra text around the JSON.
        # We try to extract the JSON part.
        try:
            json_start = response.response.find('[')
            json_end = response.response.rfind(']')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_str = response.response[json_start : json_end + 1]
                mcqs = json.loads(json_str)
            else:
                mcqs = json.loads(response.response) # Try direct parse if no brackets found
            return mcqs
        except json.JSONDecodeError as e:
            st.error(f"Failed to decode MCQs. Model returned invalid JSON. Error: {e}")
            st.code(response.response, language='json') # Show raw response for debugging
            return []

    except Exception as e:
        st.error(f"An error occurred during MCQ generation: {e}")
        return []

def query_pdf_chatbot(index: VectorStoreIndex, user_query: str):
    """
    Queries the indexed PDF content with a user's question and returns a chatbot response.
    """
    if index is None:
        st.error("PDF knowledge base not available. Please process a PDF first.")
        return "Please upload and process a PDF first."

    st.info(f"Asking chatbot: '{user_query}'")
    try:
        query_engine = index.as_query_engine()
        response = query_engine.query(user_query)
        return response.response
    except Exception as e:
        st.error(f"An error occurred during chatbot query: {e}")
        return f"Sorry, I could not process your request: {e}"

# --- Streamlit UI Layout ---

def main():
    st.set_page_config(page_title="PDF RAG Assistant", layout="wide")

    st.title("📚 PDF RAG Assistant")
    st.markdown("Upload a PDF document to generate MCQs or chat with its content.")

    # Initialize session state variables
    if 'pdf_index' not in st.session_state:
        st.session_state.pdf_index = None
    if 'current_view' not in st.session_state:
        st.session_state.current_view = 'upload' # 'upload', 'options', 'mcq', 'chatbot'
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Initialize models only once
    if not initialize_models():
        st.stop() # Stop if models can't be initialized

    # --- PDF Upload Section ---
    if st.session_state.current_view == 'upload':
        st.subheader("Step 1: Upload Your Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file is not None:
            if st.button("Process Document"):
                with st.spinner("Processing PDF..."):
                    st.session_state.pdf_index = process_pdf_and_create_index(uploaded_file)
                    if st.session_state.pdf_index:
                        st.session_state.current_view = 'options'
                        st.session_state.chat_history = [] # Reset chat history for new PDF
                        st.experimental_rerun() # Rerun to switch view

    # --- Options Section ---
    elif st.session_state.current_view == 'options':
        st.subheader("Step 2: Choose an Action")
        if st.session_state.pdf_index is None:
            st.warning("No PDF processed. Please go back to upload.")
            if st.button("Go to Upload"):
                st.session_state.current_view = 'upload'
                st.experimental_rerun()
            return

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Generate MCQs", use_container_width=True):
                st.session_state.current_view = 'mcq'
                st.experimental_rerun()
        with col2:
            if st.button("Chat with PDF", use_container_width=True):
                st.session_state.current_view = 'chatbot'
                st.experimental_rerun()

        st.markdown("---")
        if st.button("Upload New Document"):
            st.session_state.pdf_index = None
            st.session_state.current_view = 'upload'
            st.experimental_rerun()

    # --- MCQ Generator Section ---
    elif st.session_state.current_view == 'mcq':
        st.subheader("MCQ Generator")
        if st.session_state.pdf_index is None:
            st.warning("No PDF processed. Please go back to upload.")
            if st.button("Go to Upload"):
                st.session_state.current_view = 'upload'
                st.experimental_rerun()
            return

        num_questions = st.slider("Number of Questions", min_value=1, max_value=10, value=3)
        difficulty = st.selectbox("Difficulty Level", ["easy", "medium", "hard"])

        if st.button("Generate Questions"):
            with st.spinner("Generating MCQs..."):
                mcqs = generate_mcqs_from_pdf(st.session_state.pdf_index, num_questions, difficulty)
                if mcqs:
                    st.session_state.mcq_questions = mcqs # Store generated MCQs
                    st.success("MCQs generated!")
                else:
                    st.session_state.mcq_questions = [] # Clear if generation failed

        if 'mcq_questions' in st.session_state and st.session_state.mcq_questions:
            st.markdown("---")
            st.write("### Generated Multiple Choice Questions:")
            for i, q in enumerate(st.session_state.mcq_questions):
                st.markdown(f"**Q{i+1}: {q['question']}**")
                for opt_idx, option in enumerate(q['options']):
                    st.write(f"  {chr(65 + opt_idx)}. {option}")
                st.markdown(f"**Correct Answer:** {q['correctAnswer']}")
                st.markdown("---")

        if st.button("Back to Options"):
            st.session_state.current_view = 'options'
            st.experimental_rerun()

    # --- Chatbot Section ---
    elif st.session_state.current_view == 'chatbot':
        st.subheader("Chat with Your Document")
        if st.session_state.pdf_index is None:
            st.warning("No PDF processed. Please go back to upload.")
            if st.button("Go to Upload"):
                st.session_state.current_view = 'upload'
                st.experimental_rerun()
            return

        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if user_question := st.chat_input("Ask a question about the document..."):
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = query_pdf_chatbot(st.session_state.pdf_index, user_question)
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})

        if st.button("Back to Options"):
            st.session_state.current_view = 'options'
            st.experimental_rerun()

if __name__ == "__main__":
    main()
