#from langchain_community.document_loaders import PyPDFLoader

#splittext = PyPDFLoader("novel.pdf")

#docs = splittext.load()
#full_text = "\n".join([doc.page_content for doc in docs])
#print(full_text)

#from langchain_text_splitters import RecursiveCharacterTextSplitter

#text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#texts = text_splitter.split_text(full_text)




# Import the PyPDFLoader from LangChain to load PDF documents
#from langchain_community.document_loaders import PyPDFLoader

# Import CharacterTextSplitter to split large text into smaller chunks
#from langchain_text_splitters import CharacterTextSplitter


# Create a loader object and give the path to the PDF file
# This tells LangChain which PDF we want to read
#loader = PyPDFLoader("/Users/govardhank/Desktop/pythongui_test/langchain_demo/novel.pdf")


# Load the PDF file
# Each page of the PDF becomes a "Document object"
#docs = loader.load()


# Combine the text from all pages into one large string
# doc.page_content extracts the text from each page
# "\n" adds a newline between pages
#full_text = "\n".join([doc.page_content for doc in docs])


# Print the entire extracted text from the PDF
# This shows that the loader successfully read the document
#print(full_text)


# Create a text splitter object
# encoding_name="cl100k_base" → token encoding used by OpenAI models
# chunk_size=100 → each chunk will contain about 100 tokens
# chunk_overlap=0 → chunks will not overlap
#text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    #encoding_name="cl100k_base",
    #chunk_size=100,
    #chunk_overlap=0



# Split the large text into smaller chunks
# Each chunk will be about 100 tokens long
#texts = text_splitter.split_text(full_text)


# Print the number of chunks created after splitting the text
#print(len(texts))


# Print the first chunk of the split text
# This helps verify how the text has been divided
#print(texts[0])

#embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#text = "If you'd like, I can also show you the next step: converting these chunks into embeddings."
#embedding = embedder.embed_query(text)

#print(f"Embedding dimensions: {len(embedding)}")
#print(f"First 5 values: {embedding[:5]}")

#from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_community.vectorstores import FAISS

#texts = [
    #"Python is the best language for AI",
    #"This is the course for beginners",
    #"This will teach you all about GEN AI"
#]

# create embedding model
#embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# create FAISS vector database
#vectorstore = FAISS.from_texts(texts, embedder)

#print("Vector store created successfully")


#query = "Which programming language is good for AI?"

#results = vectorstore.similarity_search(query)

#print(results[0].page_content)

