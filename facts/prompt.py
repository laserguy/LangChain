from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

import langchain

# Set this to True to see what is happening behind the scenes, how langchain creates internal chains
langchain.debug = True

load_dotenv()

chat = ChatOpenAI()
embeddings = OpenAIEmbeddings()

# Load the existing Chroma DB instance (created in main.py)
db = Chroma(persist_directory="emb", embedding_function=embeddings)

"""
    Ok now we have the embeddings information of the chunks, now we want to ask our question and get the answer?
    
    Steps:
    1 - fact_chunk = Get the most relevant chunk based on embeddings similarity with question.
    2 - Create a prompt chain with input_variables=['fact_chunk','question']
    3 - Prompt will look like this
        prompt = PromptTemplate(
            input_variables=["fact_chunk", "question"],
            template="Based on the fact provided in the triple backticks
                ```{fact_chunk}```, answer the following question: '{question}' ",
        )
    
    But LangChain already contains something that does all the above automatically
    - 'RetrievalQA' does all the above steps
    
    So what is a Retriever?
        To be a "Retriever", the object must have a method called "get_relevant_documents" that takes
        s string and returns a list of documents.
"""

retriever = db.as_retriever()

"""
    db.similarity_search  => We saw in main.py, that returns the list of documents
    Flow:
        as_retriever() => retriever => get_relevant_documents(string) => similarity_search(string)
        See the diagram in video number 38 timestamp 3:05 in case any confusion
"""

# chain_type='stuff' means get relevant info out of the vector database and stuff it into the prompt
# There are other chain_types as well:
"""
    map_reduce => finds top 4 relevants documents(chunks), and creats prompt chain for each of them.
        The results of all these 4 prompt chains are combined into a single to create another prompt
        with 'question', answer from this prompt chain is the final answer.
        
    map_rerank => Almost same as above, except for each 4 prompts a score is calculated, the result
        with the highest score is returned
        
    refine => Check video number 39
"""
chain = RetrievalQA.from_chain_type(llm=chat, retriever=retriever, chain_type="stuff")

result = chain.run("What is an interesting fact about the english language?")

print(result)
