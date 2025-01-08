-----> Medical Chatbot <------

project name: medical chatbot

workes/components/operatons
steps: 
1. data_ingestion.py:
                contains:
                        -loading the data pdf from medical book
                        -splitting them into the chunks
                        -open source huggingface embedding model is used
                        -FAISS vector data base is used to store the embedded data.
                        -save them in locally

2. main.py:
        contains:
                -load vector from the local
                -Open source LLM model is used to text generation.
                -LLM model calls through huggingface Endpoint.
                -streamlit is used for UI and chat histry container.

3.pra.ipynb:
        used for practice then the same code converted into the moduler codding.


4. Used libraries/frameworks:
                        -langchain
                        -huggingface
                        -FAISS
                        -streamlit


5. Requirements:
            -Huggingface api_key
            -and all packages and libraries mentioned in the requirements.txt file.



NOTE:
    1.Asked any think related to medical and diagnostic purposes.
    2."sentence-transformers/all-mpnet-base-v2". this model is used in code for embedding
    3."microsoft/Phi-3.5-mini-instruct". LLM Model used for text generation.

                -