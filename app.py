import os
import streamlit as st
import re
import json
from datetime import datetime, timedelta
import pytz
from docx import Document  # For .docx
import pandas as pd
from bs4 import BeautifulSoup
import json
import yaml
from pptx import Presentation
from zipfile import ZipFile
import mimetypes 
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from supabase import create_client, Client

# Supabase credentials
# Supabase credentials
url = os.getenv("url")
key = os.getenv("key")
supabase: Client = create_client(url, key)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Load the model
@st.cache_resource
def load_model():
    return ChatGroq(temperature=0.8, model="llama3-8b-8192")

def clean_text(text):
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Fix broken sentences or words caused by line breaks
    text = re.sub(r'(?<=[a-zA-Z])\s(?=[a-zA-Z])', '', text)  # Remove single spaces in the middle of words

    # Standardize newlines for better formatting
    text = re.sub(r'\.\s+', '.\n', text)  # Add newlines after sentences
    text = re.sub(r'(?<=:)\s+', '\n', text)  # Add newlines after colons
    
    # Additional cleanup (if needed)
    text = text.strip()  # Remove leading and trailing whitespace

    return text

@st.cache_data
def load_hidden_documents(bucket_name="hidden_docs"):
    """Load all supported file types from the hidden_docs bucket and return their content."""
    all_texts = []

    # List files in the hidden_docs bucket
    #response = supabase.storage.from_(bucket_name).list()
    response = supabase.storage.from_(bucket_name).list(  "folder",
                                                          {"limit": 100, "offset": 0, "sortBy": {"column": "name", "order": "desc"}},)
    st.write(type(response))
    st.write(response)
    for file_info in response:
        file_name = file_info['name']
        file_path = file_info['name']
        st.write(file_name)
        # Retrieve the file from the bucket
        file = supabase.storage.from_(bucket_name).download(file_path)

        if file.get('status_code') == 200:
            file_content = file['data']

            mime_type, _ = mimetypes.guess_type(file_path)

            try:
                # Handle PDF files
                if file_name.endswith(".pdf"):
                    loader = PyPDFLoader(BytesIO(file_content))
                    pages = loader.load_and_split()
                    all_texts.extend([page.page_content for page in pages])

                # Handle Word files (.docx)
                elif file_name.endswith(".docx"):
                    doc = Document(BytesIO(file_content))
                    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                    all_texts.append(text)

                # Handle Text files (.txt)
                elif file_name.endswith(".txt"):
                    all_texts.append(file_content.decode("utf-8"))

                # Handle Excel files (.xlsx and .xls)
                elif file_name.endswith(('.xlsx', '.xls')):
                    excel_data = pd.read_excel(BytesIO(file_content))
                    text = excel_data.to_string(index=False)
                    all_texts.append(text)

                # Handle CSV files (.csv)
                elif file_name.endswith(".csv"):
                    csv_data = pd.read_csv(BytesIO(file_content))
                    text = csv_data.to_string(index=False)
                    all_texts.append(text)

                # Handle Markdown files (.md)
                elif file_name.endswith(".md"):
                    all_texts.append(file_content.decode("utf-8"))

                # Handle HTML files (.html, .htm)
                elif file_name.endswith(('.html', '.htm')):
                    soup = BeautifulSoup(file_content, "html.parser")
                    all_texts.append(soup.get_text())

                # Handle JSON files (.json)
                elif file_name.endswith(".json"):
                    data = json.loads(file_content.decode("utf-8"))
                    all_texts.append(json.dumps(data, indent=2))

                # Handle YAML files (.yaml, .yml)
                elif file_name.endswith(('.yaml', '.yml')):
                    data = yaml.safe_load(file_content.decode("utf-8"))
                    all_texts.append(json.dumps(data, indent=2))

                # Handle PowerPoint files (.pptx)
                elif file_name.endswith(".pptx"):
                    presentation = Presentation(BytesIO(file_content))
                    for slide in presentation.slides:
                        slide_text = []
                        for shape in slide.shapes:
                            if shape.has_text_frame:
                                slide_text.append(shape.text)
                        all_texts.append("\n".join(slide_text))

                # Handle ZIP files (.zip)
                elif file_name.endswith(".zip"):
                    with zipfile.ZipFile(BytesIO(file_content), 'r') as zip_ref:
                        zip_ref.extractall("temp_extracted")
                        all_texts.extend(load_hidden_documents("temp_extracted"))

                # Handle Log files (.log)
                elif file_name.endswith(".log"):
                    all_texts.append(file_content.decode("utf-8"))

                # Handle unknown file types (fallback to text-based reading)
                elif mime_type and mime_type.startswith("text"):
                    all_texts.append(file_content.decode("utf-8"))

            except Exception as e:
                print(f"Failed to process {file_name}: {e}")
        else:
            print(f"Failed to retrieve file: {file_name}")

    # Clean and return all the extracted texts
    cleaned_texts = [clean_text(text) for text in all_texts]
    return cleaned_texts


@st.cache_data
def save_to_supabase(all_texts):
    """Save the list of documents to the Supabase 'all_texts' table."""
    for text in all_texts:
        data = {"all_texts": text}
        response = supabase.table("all_texts").insert(data).execute()
        
        # Check the response for success or failure
        if response.data:  # If the response contains data, the insert was successful
            print(f"Successfully saved: {text[:30]}...")
        else:  # If there is an error
            print(f"Failed to save text")

# Load documents and save to Supabase
all_texts = load_hidden_documents()
st.write(f"Text generated from documents is {all_texts[0:200]}")
save_to_supabase(all_texts)

# Create vector store
@st.cache_resource
def create_vector_store(document_texts):
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(document_texts, embedder)

# Get modification times for documents
@st.cache_data
def get_file_mod_times(directory):
    """Get the modification times of all files in the directory."""
    return {
        f: os.path.getmtime(os.path.join(directory, f))
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))  # Ensure it's a file, not a directory
    }

# Reload vector store if needed
def reload_vector_store_if_needed():
    current_mod_times = get_file_mod_times("hidden_docs")

    # Check if file modifications have occurred
    if "file_mod_times" not in st.session_state or st.session_state["file_mod_times"] != current_mod_times:
        st.session_state["file_mod_times"] = current_mod_times
        document_texts = load_hidden_documents()
        vector_store = create_vector_store(document_texts)  # Create vector store
        st.session_state["vector_store"] = vector_store    # Save in session state
    else:
        # Retrieve from session state if already initialized
        vector_store = st.session_state.get("vector_store", None)

    # Return the vector store (even if None)
    return vector_store

# Load model and vector store
model = load_model()

# Initialize vector_store
vector_store = reload_vector_store_if_needed()

# If still None, raise an error to debug initialization
if vector_store is None:
    raise ValueError("Failed to initialize vector_store. Ensure hidden_docs folder and embeddings setup are correct.")
##### my email lalitmach22@gmail.com
# Validate email
def is_valid_email(email):
    email_regex = r"^\d{2}f\d{7}@ds\.study\.iitm\.ac\.in$"
    return re.match(email_regex, email) is not None or email == "nitin@ee.iitm.ac.in"

# Save session to Supabase
def save_session_to_supabase(email, name, chat_history):
    # Define IST timezone
    ist = pytz.timezone("Asia/Kolkata")
    
    for question, answer in chat_history:
        # Get current datetime in IST and format as "YYYY-MM-DD HH:MM"
        timestamp = datetime.now(ist).strftime("%Y-%m-%d %H:%M")
        
        data = {
            "email": email,
            "name": name if name else None,
            "question": question,
            "answer": answer,
            "timestamp": timestamp,  # Add IST timestamp
        }
        response = supabase.table("chat_sessions").insert(data).execute()
        if "error" in response:
            st.error(f"Error saving session data to Supabase: {response['error']['message']}")
            return False
    return True


# Streamlit app
st.title("BDM Chatbot")
st.write("Developed by Lalit & Puneet of BS, IIT Madras")
st.write("Ask questions directly based on the preloaded BDM documents.")
st.write("Note - Once your queries are complete, please put the last query as \"stop\".")
st.write("Disclaimer - All data, including questions and answers, is collected for improving the bot’s functionality. By using this bot, you consent to this data being stored.")
st.write("Disclaimer - This chatbot can make mistakes. For verified and detailed information, please refer to the official project documents or consult the relevant team members/mentors.")
# Load model and vector store
model = load_model()
vector_store = reload_vector_store_if_needed()
retrieval_chain = ConversationalRetrievalChain.from_llm(model, retriever=vector_store.as_retriever())

# Initialize session states
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "email_validated" not in st.session_state:
    st.session_state["email_validated"] = False
if "session_start_time" not in st.session_state:
    st.session_state["session_start_time"] = datetime.now()

# Email input
email = st.text_input("Enter your email (format: XXfXXXXXXX@ds.study.iitm.ac.in):")
name = st.text_input("Enter your name (optional):")

if email and is_valid_email(email):
    st.session_state["email_validated"] = True
    st.success("Email validated successfully! You can now ask your questions.")
elif email:
    st.error("Invalid email format. Please enter a valid email.")

# Check session time limit
elapsed_time = datetime.now() - st.session_state["session_start_time"]
if elapsed_time > timedelta(minutes=30):
    st.warning("Session time has exceeded 30 minutes. The chatbot will stop now.")
    
    # Save session data to Supabase
    if save_session_to_supabase(email, name, st.session_state["chat_history"]):
        st.success("Session data successfully saved to Supabase!")
    
    # Download session data
    session_data = {
    "email": email,
    "name": name,
    "chat_history": [
        {"question": q, "answer": a, "timestamp": datetime.now().isoformat()}
        for q, a in st.session_state["chat_history"]
    ]
}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"session_data_{timestamp}.json"
    st.download_button(
        label="Download Session Data",
        data=json.dumps(session_data, indent=4),
        file_name=filename,
        mime="application/json"
    )
    st.stop()

# Chat functionality
if st.session_state["email_validated"]:
    user_input = st.text_input("Pose your Questions:")
    
    if user_input:
        if user_input.lower() == "stop":
            st.write("Chatbot: Goodbye!")
            session_data = {
                "email": email,
                "name": name,
                "chat_history": st.session_state["chat_history"]
            }

            # Save session data to Supabase
            if save_session_to_supabase(email, name, st.session_state["chat_history"]):
                st.success("Session data successfully saved to Supabase!")

            # Allow download of session data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"session_data_{timestamp}.json"
            st.download_button(
                label="Download Session Data",
                data=json.dumps(session_data, indent=4),
                file_name=filename,
                mime="application/json"
            )
            st.stop()
        else:
            # Process the question and answer
            response = retrieval_chain.invoke({"question": user_input, "chat_history": st.session_state["chat_history"]})
            answer = response["answer"]
            st.session_state["chat_history"].append((user_input, answer))
            
            # Display the chat history
            # Display the chat history (most recent first)
            for i, (question, reply) in enumerate(reversed(st.session_state["chat_history"]), 1):
                st.write(f"Q{i}: {question}")
                st.write(f"Chatbot: {reply}")

