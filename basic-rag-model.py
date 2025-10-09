import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# 1. Define the source(s) to load
# We provide a list of URLs, even if it's just one
urls = ("https://lilianweng.github.io/posts/2023-06-23-agent/",)

# Optional: Use bs_kwargs to filter specific parts of the page
# This helps extract only the main content you care about
loader = WebBaseLoader(
    web_paths=urls,
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

# 2. Load the documents
docs = loader.load()

# Let's look at the first loaded document
first_document = docs[0]

# # Print the main content
# print("--- Page Content Preview ---")
# print(first_document.page_content[:500] + "...") # Print only the first 500 characters

# # Print the metadata
# print("\n--- Metadata ---")
# print(first_document.metadata)









from langchain.text_splitter import RecursiveCharacterTextSplitter

# Assume 'docs' is the list of Document objects loaded in Chapter 3

# 1. Define the splitter
# We set a chunk size (e.g., 1000 characters)
# and a chunk overlap (e.g., 200 characters)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# 2. Split the documents
splits = text_splitter.split_documents(docs)

# 'splits' is now a list of smaller Document objects
# Each object contains a chunk of the original text
# and retains the metadata from the original document.


# # Print the number of chunks created
# print(f"Original documents: {len(docs)}")
# print(f"Number of chunks: {len(splits)}")

# # Print the content of the first chunk
# print("\n--- First Chunk ---")
# print(splits[0].page_content)

# # Print the content of the second chunk
# print("\n--- Second Chunk ---")
# print(splits[1].page_content)


# Notice the overlap between the end of the first chunk and the beginning of the second!

# Initialize the embedding model
# model="text-embedding-ada-002" is a common and cost-effective model
# You can try others like "nomic-embed-text" from nomic-ai (requires different package)
# For this tutorial, we'll stick with OpenAIEmbeddings for consistency
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Embed the text chunks
vectors = embeddings_model.embed_documents([s.page_content for s in splits])

# Print the shape of the vectors
print(f"Shape of the vectors: {vectors[0].shape}")
