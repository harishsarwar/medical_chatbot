import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
import os

# Environment setup
repo_id = "microsoft/Phi-3.5-mini-instruct"
api_key = os.getenv("HUGGINGFACE_API_KEY")
os.environ["HUGGINGFACE_API_KEY"] = api_key

DB_FAISS_PATH = "vectorstores/db_faiss"
embedding_model = "sentence-transformers/all-mpnet-base-v2"  # Embedding model for chunk embeddings

# Load the vector database
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={'device': 'cpu'})
    return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

# Chatbot logic using vector search and Hugging Face model
def get_chatbot_response(query):
    db = load_vector_db()
    docs = db.similarity_search(query, k=1)

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

# Conversational chat function to manage user input and chatbot response
def conversational_chat(query):
    result = get_chatbot_response(query)
    st.session_state['history'].append((query, result))
    return result

# Streamlit app for chatting
def chatbot_app():
    st.title("Medical Chatbot")

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about the medical."]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

    response_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Ask a question:", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversational_chat(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                st.markdown(f"**User:** {st.session_state['past'][i]}")
                st.markdown(f"**Bot:** {st.session_state['generated'][i]}")

if __name__ == "__main__":
    chatbot_app()
