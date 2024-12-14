import os
import streamlit as st
import re
import json
from datetime import datetime, timedelta
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from supabase import create_client, Client

# Supabase credentials
url = "https://armzsxwnhybsgedffijs.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InljcXF6b3NsdXlocWR3dG5yaWNyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzQxODE5NjksImV4cCI6MjA0OTc1Nzk2OX0.Q-AmMeWuUkODkX27tg7YEN9bLvqz7v8qOHqpIucs_iw"
supabase: Client = create_client(url, key)

os.environ["GROQ_API_KEY"] = "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"

# Load the model
@st.cache_resource
def load_model():
    return ChatGroq(temperature=0.8, model="llama3-8b-8192")

# Load PDF files
@st.cache_data
def load_hidden_pdfs(directory="hidden_docs"):
    all_texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(directory, filename))
            pages = loader.load_and_split()
            all_texts.extend([page.page_content for page in pages])
    return all_texts

# Create vector store
@st.cache_resource
def create_vector_store(document_texts):
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(document_texts, embedder)

# Get modification times for documents
@st.cache_data
def get_file_mod_times(directory):
    """Get the modification times of all files in the directory."""
    return {f: os.path.getmtime(os.path.join(directory, f)) for f in os.listdir(directory) if f.endswith(".pdf")}

# Reload vector store if needed
def reload_vector_store_if_needed():
    current_mod_times = get_file_mod_times("hidden_docs")

    # Check if file modifications have occurred
    if "file_mod_times" not in st.session_state or st.session_state["file_mod_times"] != current_mod_times:
        st.session_state["file_mod_times"] = current_mod_times
        document_texts = load_hidden_pdfs()
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
#####LALIT lalitmach22@gmail.com
# Validate email
def is_valid_email(email):
    email_regex = r"^\d{2}f\d{7}@ds\.study\.iitm\.ac\.in$"
    return re.match(email_regex, email) is not None or email == "nitin@ee.iitm.ac.in"

# Save session to Supabase
def save_session_to_supabase(email, name, chat_history):
    for question, answer in chat_history:
        data = {
            "email": email,
            "name": name if name else None,
            "question": question,
            "answer": answer,
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
        "chat_history": st.session_state["chat_history"]
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
            for i, (question, reply) in enumerate(st.session_state["chat_history"], 1):
                st.write(f"Q{i}: {question}")
                st.write(f"Chatbot: {reply}")
