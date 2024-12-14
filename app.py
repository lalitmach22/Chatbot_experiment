# version 1 

# import os
# import streamlit as st
# from langchain_groq import ChatGroq
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.chains import ConversationalRetrievalChain
# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings

# os.environ["GROQ_API_KEY"] = "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"

# @st.cache_resource
# def load_model():
#     return ChatGroq(temperature=0.8, model="llama3-8b-8192")

# @st.cache_data
# def load_hidden_pdfs(directory="hidden_docs"):
#     all_texts = []
#     for filename in os.listdir(directory):
#         if filename.endswith(".pdf"):
#             loader = PyPDFLoader(os.path.join(directory, filename))
#             pages = loader.load_and_split()
#             all_texts.extend([page.page_content for page in pages])
#     return all_texts

# @st.cache_resource
# def create_vector_store(document_texts):
#     embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     return FAISS.from_texts(document_texts, embedder)

# st.title("BDM Chatbot")
# st.write("Ask questions directly based on the preloaded BDM documents.")

# model = load_model()
# document_texts = load_hidden_pdfs()
# vector_store = create_vector_store(document_texts)
# retrieval_chain = ConversationalRetrievalChain.from_llm(model, retriever=vector_store.as_retriever())
    
# if "chat_history" not in st.session_state:
#     st.session_state["chat_history"] = []
        
# user_input = st.text_input("Pose your Questions:")

# if user_input:
#     if user_input.lower() == "stop":
#         st.write("Chatbot: Goodbye!")
#         st.stop()
#     else:
#         response = retrieval_chain.invoke({"question": user_input, "chat_history": st.session_state["chat_history"]})
#         answer = response["answer"]
#         st.session_state["chat_history"].append((user_input, answer))
#         for i, (question, reply) in enumerate(st.session_state["chat_history"], 1):
#             st.write(f"Q{i}: {question}")
#             st.write(f"Chatbot: {reply}")

# version 2 - without custom embeddings 

# import os
# import streamlit as st
# import re
# import json
# from datetime import datetime
# from langchain_groq import ChatGroq
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.chains import ConversationalRetrievalChain
# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings

# os.environ["GROQ_API_KEY"] = "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"

# @st.cache_resource
# def load_model():
#     return ChatGroq(temperature=0.8, model="llama3-8b-8192")

# @st.cache_data
# def load_hidden_pdfs(directory="hidden_docs"):
#     all_texts = []
#     for filename in os.listdir(directory):
#         if filename.endswith(".pdf"):
#             loader = PyPDFLoader(os.path.join(directory, filename))
#             pages = loader.load_and_split()
#             all_texts.extend([page.page_content for page in pages])
#     return all_texts

# @st.cache_resource
# def create_vector_store(document_texts):
#     embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     return FAISS.from_texts(document_texts, embedder)

# def is_valid_email(email):
#     email_regex = r"^\d{2}f\d{7}@ds\.study\.iitm\.ac\.in$"
#     return re.match(email_regex, email) is not None

# st.title("BDM Chatbot")
# st.write("Ask questions directly based on the preloaded BDM documents.")

# model = load_model()
# document_texts = load_hidden_pdfs()
# vector_store = create_vector_store(document_texts)
# retrieval_chain = ConversationalRetrievalChain.from_llm(model, retriever=vector_store.as_retriever())

# if "chat_history" not in st.session_state:
#     st.session_state["chat_history"] = []
# if "email_validated" not in st.session_state:
#     st.session_state["email_validated"] = False

# email = st.text_input("Enter your email (format: XXfXXXXXXX@ds.study.iitm.ac.in):")
# name = st.text_input("Enter your name (optional):")

# if email and is_valid_email(email):
#     st.session_state["email_validated"] = True
#     st.success("Email validated successfully! You can now ask your questions.")
# elif email:
#     st.error("Invalid email format. Please enter a valid email.")
# if st.session_state["email_validated"]:
#     user_input = st.text_input("Pose your Questions:")
    
#     if user_input:
#         if user_input.lower() == "stop":
#             st.write("Chatbot: Goodbye!")
#             session_data = {
#                 "email": email,
#                 "name": name,
#                 "chat_history": st.session_state["chat_history"]
#             }
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             filename = f"session_data_{timestamp}.json"
#             st.download_button(
#                 label="Download Session Data",
#                 data=json.dumps(session_data, indent=4),
#                 file_name=filename,
#                 mime="application/json"
#             )
#             st.stop()
#         else:
#             response = retrieval_chain.invoke({"question": user_input, "chat_history": st.session_state["chat_history"]})
#             answer = response["answer"]
#             st.session_state["chat_history"].append((user_input, answer))

#             for i, (question, reply) in enumerate(st.session_state["chat_history"], 1):
#                 st.write(f"Q{i}: {question}")
#                 st.write(f"Chatbot: {reply}")

# version 3 - trying supabase

import os
import streamlit as st
import re
import json
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from supabase import create_client, Client

url = "https://armzsxwnhybsgedffijs.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFybXpzeHduaHlic2dlZGZmaWpzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzMwODcxMzEsImV4cCI6MjA0ODY2MzEzMX0.g7Ty0qNFCVJiEp38IQ_Uw9yEn4jzA67XPsLCmQ8f26o"
supabase: Client = create_client(url, key)

os.environ["GROQ_API_KEY"] = "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"

@st.cache_resource
def load_model():
    return ChatGroq(temperature=0.8, model="llama3-8b-8192")

@st.cache_data
def load_hidden_pdfs(directory="hidden_docs"):
    all_texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(directory, filename))
            pages = loader.load_and_split()
            all_texts.extend([page.page_content for page in pages])
    return all_texts

@st.cache_resource
def create_vector_store(document_texts):
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(document_texts, embedder)

def is_valid_email(email):
    email_regex = r"^\d{2}f\d{7}@ds\.study\.iitm\.ac\.in$"
    return re.match(email_regex, email) is not None or email == "nitin@ee.iitm.ac.in"

# Save session data (question and answer pairs) to Supabase
def save_session_to_supabase(email, name, chat_history):
    for question, answer in chat_history:
        data = {
            "email": email,
            "name": name if name else None,
            "question": question,
            "answer": answer,
        }
        # Insert data into Supabase
        response = supabase.table("chat_sessions").insert(data).execute()

        # Check if response has errors (if any)
        if "error" in response:
            st.error(f"Error saving session data to Supabase: {response['error']['message']}")
            return False
    return True

st.title("BDM Chatbot")
st.write("Ask questions directly based on the preloaded BDM documents.")
st.write("Note - Once your queries are complete, please put the last query as \"stop\".")
st.write("Disclaimer - All data, including questions and answers, is collected for improving the bot’s functionality. By using this bot, you consent to this data being stored.")

model = load_model()
document_texts = load_hidden_pdfs()
vector_store = create_vector_store(document_texts)
retrieval_chain = ConversationalRetrievalChain.from_llm(model, retriever=vector_store.as_retriever())

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "email_validated" not in st.session_state:
    st.session_state["email_validated"] = False

email = st.text_input("Enter your email (format: XXfXXXXXXX@ds.study.iitm.ac.in):")
name = st.text_input("Enter your name (optional):")

if email and is_valid_email(email):
    st.session_state["email_validated"] = True
    st.success("Email validated successfully! You can now ask your questions.")
elif email:
    st.error("Invalid email format. Please enter a valid email.")

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

            # Allow the user to download session data as JSON when they say "stop"
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


# version 5 - added download feature - with custom embeedings that doesn't work

# import os
# import re
# import json
# import streamlit as st
# from datetime import datetime
# from langchain_groq import ChatGroq
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.chains import ConversationalRetrievalChain
# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings

# # Your API Key (ensure it's handled securely, not hardcoded in production)
# os.environ["GROQ_API_KEY"] = "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"

# @st.cache_resource
# def load_model():
#     return ChatGroq(temperature=0.8, model="llama3-8b-8192")

# @st.cache_data
# def load_hidden_pdfs(directory="hidden_docs"):
#     all_texts = []
#     for filename in os.listdir(directory):
#         if filename.endswith(".pdf"):
#             loader = PyPDFLoader(os.path.join(directory, filename))
#             pages = loader.load_and_split()
#             all_texts.extend([page.page_content for page in pages])
#     return all_texts

# @st.cache_resource
# def create_and_save_vector_store(document_texts, save_path="faiss_index"):
#     embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vector_store = FAISS.from_texts(document_texts, embedder)
#     vector_store.save_local(save_path)
#     return vector_store

# @st.cache_resource
# def load_vector_store(save_path="faiss_index"):
#     if os.path.exists(save_path):
#         embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#         return FAISS.load_local(save_path, embedder)
#     return None

# @st.cache_resource
# def get_vector_store(document_texts, save_path="faiss_index"):
#     if os.path.exists(save_path):
#         st.write("Loading existing FAISS index...")
#         return load_vector_store(save_path)
#     else:
#         st.write("Creating and saving a new FAISS index...")
#         return create_and_save_vector_store(document_texts, save_path)

# def is_valid_email(email):
#     email_pattern = r"^\d{2}f\d{7}@ds\.study\.iitm\.ac\.in$"
#     return bool(re.match(email_pattern, email))

# def save_session_data(email, name, questions_and_answers):
#     # Create timestamped filename
#     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     session_data = {
#         "email": email,
#         "name": name,
#         "timestamp": timestamp,
#         "questions_and_answers": questions_and_answers
#     }
    
#     save_path = f"session_data_{timestamp}.json"
#     with open(save_path, "w") as file:
#         json.dump(session_data, file, indent=4)

#     return save_path

# st.title("BDM Chatbot")
# st.write("Ask questions directly based on the preloaded BDM documents.")

# email = st.text_input("Enter your email ID:")
# name = st.text_input("Enter your name (optional):")

# if email:
#     if is_valid_email(email):
#         st.session_state['email_valid'] = True
#         st.write("Email is valid! Now you can ask your questions.")

#         model = load_model()
#         document_texts = load_hidden_pdfs()

#         vector_store = get_vector_store(document_texts)

#         retrieval_chain = ConversationalRetrievalChain.from_llm(model, retriever=vector_store.as_retriever())

#         if "chat_history" not in st.session_state:
#             st.session_state["chat_history"] = []

#         user_input = st.text_input("Pose your Questions:")

#         if user_input:
#             if user_input.lower() == "stop":
#                 st.write("Chatbot: Goodbye!")
#                 # Save session data with timestamped filename
#                 session_file = save_session_data(email, name, st.session_state["chat_history"])

#                 # Provide download link for session data
#                 with open(session_file, "rb") as file:
#                     st.download_button("Download Session Data", file, file_name=session_file)

#                 st.session_state["chat_history"] = []  # Reset chat history after saving
#                 st.stop()  # End the app session
                
#             else:
#                 response = retrieval_chain.invoke({"question": user_input, "chat_history": st.session_state["chat_history"]})
#                 answer = response["answer"]
#                 st.session_state["chat_history"].append((user_input, answer))
#                 for i, (question, reply) in enumerate(st.session_state["chat_history"], 1):
#                     st.write(f"Q{i}: {question}")
#                     st.write(f"Chatbot: {reply}")

#     else:
#         st.write("Invalid email. Please use the format: XXfXXXXXXX@ds.study.iitm.ac.in")

# else:
#     st.write("Please enter your email ID to proceed.")
