import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from langchain.chat_models import ChatOpenAI  # Corrected import
from langchain.schema import HumanMessage
import PyPDF2
import json
import os

# Import the tokenizer and define the count_tokens function
from transformers import GPT2TokenizerFast

# Initialize the tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text):
    """Function to count the number of tokens in a text string."""
    return len(tokenizer.encode(text))

# Initialize the model
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# Initialize ChromaDB client with the correct parameter
chroma_client = chromadb.Client(Settings(persist_directory=".chroma_db"))

# Create or get a collection
collection = chroma_client.get_or_create_collection(name="my_collection")

# Clear the collection (optional)
# Uncomment the following line to clear the collection before adding new documents
# collection.delete()

# Load documents
def load_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        text = ""
        if filename.endswith('.pdf'):
            with open(file_path, 'rb') as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                text = ''.join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif filename.endswith('.json'):
            with open(file_path) as json_file:
                data = json.load(json_file)
                text = json.dumps(data)
        if text:
            documents.append({"content": text, "filename": filename})
    return documents

documents = load_documents("ESRS")

# Embed and store in VectorDB
for doc in documents:
    embedding = model.encode(doc['content'])
    collection.add(
        documents=[doc['content']],
        embeddings=[embedding],
        metadatas=[{"filename": doc['filename']}],
        ids=[doc['filename']]
    )

# Streamlit UI
st.title("Sustainability Reporting Assistant")
st.write("Ask questions about your company's sustainability reporting based on ESRS standards.")

query = st.text_input("Enter your question:")

if query:
    # Retrieve relevant chunks
    query_embedding = model.encode(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,  # Adjust as needed
        include=['documents', 'metadatas', 'distances']
    )

    # Prepare the context from the retrieved chunks
    contexts = results['documents'][0]
    # Limit the total context length
    max_context_length = 15000  # Adjust as needed
    context = ""
    for text in contexts:
        if len(context) + len(text) <= max_context_length:
            context += text + "\n\n"
        else:
            break

    # Build the prompt, tailored to sustainability reporting
    prompt = f"""
    ## Prompt:
    **Task:** Answer the user's query based on the provided sustainability reports.
    **Query:** {query}
    **Context:**
    {context}

    **Guidelines:**
    - Ensure your response is accurate and relevant to the query.
    - Reference specific sections of the sustainability reports if possible.
    - Consider the following sustainability reporting standards: GRI, SASB, TCFD.
    - If the query is unclear or the information is not available, provide a polite and informative response.
    """

    # Estimate tokens
    total_tokens = count_tokens(prompt)
    if total_tokens > 8000:
        st.error(f"The prompt is too long ({total_tokens} tokens). Please reduce the input size.")
        st.stop()

    # Initialize the ChatOpenAI LLM
    api_key = "" # Use environment variable for API key


    llm = ChatOpenAI(
        openai_api_key=api_key,
        model_name="gpt-4",
        temperature=0.4,
        max_tokens=500  # Adjust as needed
    )

    # Create the messages for the chat
    messages = [HumanMessage(content=prompt)]

    # Generate the answer with error handling
    try:
        response = llm(messages)
        answer = response.content
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.stop()

    # Display the answer
    st.write("### Answer")
    st.write(answer)
