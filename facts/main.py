from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
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

for doc in docs:
    print(doc.page_content)
    print("\n")
