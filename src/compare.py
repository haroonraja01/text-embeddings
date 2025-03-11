from langchain_community.document_loaders import TextLoader
# from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
import os
from sentence_transformers import SentenceTransformer
# Load the document, split it into chunks, embed each chunk and load it into the vector store.
# raw_documents = TextLoader('state_of_the_union.txt').load()
# raw_documents = reader.pages[0].extract_text()

from langchain_community.document_loaders import PyPDFLoader


datadir = "/Users/haroonraja/Google Drive/Colab Notebooks/QMS/dataset"
loader = PyPDFLoader(os.path.join(datadir, "tmlr.pdf"))
pages = []
for page in loader.load():
    pages.append(page)
text_splitter = CharacterTextSplitter(chunk_size=240, chunk_overlap=0)
documents = text_splitter.split_documents(pages)

loader2 = PyPDFLoader(os.path.join(datadir, "proceedings.pdf"))
pages2 = []
for page in loader2.load():
    pages2.append(page)
# text_splitter = CharacterTextSplitter(chunk_size=240, chunk_overlap=0)
documents2 = text_splitter.split_documents(pages2)
print(documents[0])

# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"torch_dtype": "float16"})

sentences = [
    "That is a happy person",
    "That is a happy dog",
    "That is a very happy person",
    "Today is a sunny day"
]
sentences = [
    documents[0].page_content,
    documents[1].page_content,
    documents[2].page_content,
    documents[3].page_content
]
embeddings = model.encode(sentences)

sentences2 = [
    documents2[0].page_content,
    documents2[1].page_content,
    documents2[2].page_content,
    documents2[3].page_content
]
embeddings2 = model.encode(sentences2)
similarities = model.similarity(embeddings, embeddings2)
print(similarities.shape)
print(similarities)