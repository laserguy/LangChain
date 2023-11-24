from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

"""
    It is better to divide the file into chunks, so that it becomes easy to create prompts with specific
    chunks instead of the whole file.
    chunk_size => minimum length of the chunk,
        This is how this works(here):
            First 200 characters are found, then next separator character is looked for
            chunk size = 200 + characters until next '\n' is found
"""
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=200, chunk_overlap=0)

loader = TextLoader("facts.txt")
docs = loader.load_and_split(
    text_splitter=text_splitter
)  # Creates a document class from the unstructured data stored in files


"""
    Create a chroma DB instance and store all the embeddings of chunked data
    ChromaDB is vector DB, to store word vectors, can be used separately "pip install chromadb"
    ChromaDB is linked with the SQLite for the storage, can be used separately but
    Here Chroma DB is being interfaced with the Langchain (as langchain has integrated with many
    things to make the development process easier)
"""

db = Chroma.from_documents(
    docs, embedding=embeddings, persist_directory="emb"
)  # Calculate embeddings of all the docs and create a chroma DB instance, that will persist it in 'emb' directory


# Find the chunk which has highest similarity with the quoted("") text, return k chunks (again using embeddings)
results = db.similarity_search_with_score(
    "What is an interesting fact about the english language?", k=4
)

# Printing the results recieved
for result in results:
    print("\n")
    print(result[1])
    print(result[0].page_content)
