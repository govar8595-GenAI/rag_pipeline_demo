# Import embedding model from HuggingFace
from langchain_huggingface import HuggingFaceEmbeddings

# Import Chroma vector database
from langchain_community.vectorstores import Chroma


# Sample text data (documents)
texts = [
    "Artificial intelligence is transforming industries around the world.",
    "Machine learning helps computers learn patterns from data.",
    "Python is widely used for data science and AI development.",
    "Large language models can understand and generate human-like text.",
    "Vector databases allow efficient similarity search using embeddings."
]


# Load the embedding model
# This converts text into vectors (numbers)
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Create Chroma vector database
vectorstore = Chroma.from_texts(
    texts,
    embedder,
    persist_directory="chroma_db"  # folder where vectors will be stored
)


# Save the database
vectorstore.persist()


print("Chroma vector database created successfully!")


# Test similarity search
query = "what is vector database?"

results = vectorstore.similarity_search(query)


print("\nQuery Result:")
print(results[0].page_content)
