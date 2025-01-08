import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
import os

repo_id = "microsoft/Phi-3.5-mini-instruct"
# Set environment variables for HuggingFace API key
api_key = os.getenv("HUGGINGFACE_API_KEY")
os.environ["HUGGINGFACE_API_KEY"] = api_key

# Paths for the FAISS database
DB_FAISS_PATH = "vectorstores/db_faiss"
embedding_model = "sentence-transformers/all-mpnet-base-v2"  # Embedding model for chunk embeddings

# Load vector database created from data_ingestion.py
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={'device': 'cpu'})
    return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

# Chatbot logic using vector search and Hugging Face model
def get_chatbot_response(query):
    db = load_vector_db()

    # Perform similarity search in the database
    docs = db.similarity_search(query, k=1)

    # If no relevant documents found, return a fallback response
    if not docs:
        return "Sorry, I can't help with that."

    # Use Hugging Face model for response generation
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_new_tokens=100, 
        temperature=0.7,
        do_sample=True,
        repetition_penalty=1.1,
        return_full_text=True,
        huggingfacehub_api_token=api_key
    )
    response = llm.invoke(docs[0].page_content + "\n\n" + query)
    
    return response

# Streamlit app for chatting
def chatbot_app():
    st.title("Medical Book Chatbot")

    # Initialize chat history session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Input for user question
    user_input = st.text_input("Ask a question:")

    if user_input:
        # Get the chatbot response
        response = get_chatbot_response(user_input)

        # Store the conversation in the session state
        st.session_state.chat_history.append(("User", user_input))
        st.session_state.chat_history.append(("Bot", response))

    # Display chat history
    if st.session_state.chat_history:
        for role, message in st.session_state.chat_history:
            st.markdown(f"**{role}:** {message}")

if __name__ == "__main__":
    # Run the chatbot app
    chatbot_app()
