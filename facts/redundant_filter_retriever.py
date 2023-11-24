"""
    The purpose of this file:
    First if you want to know retriever, then check 'prompt.py' or check internet
    
    Lets say that the file/book/pdf etc, have many occurence of the same texts, then we would find many
    similar embeddings, and we don't want to return multiple similar embeddings
    
    `EmbeddingsRedundantFilter`, can be used to to filter, but if we see the code in the prompt.py
    it doesn't fit, more information can be found in the video 40 about why we had to create our own
    filter
"""

from langchain.embeddings.base import Embeddings
from langchain.vectorstores.chroma import Chroma
from langchain.schema import BaseRetriever


class RedundantFilterRetriever(BaseRetriever):
    embeddings: Embeddings
    chroma: Chroma

    # Following two functions have to be implemented to extend the BaseRetriever
    def get_relevant_documents(self, query):
        emb = self.embeddings.embed_documents(query)
        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding=emb,
            lambda_mult=0.8,  # Similarity upto 0.8 between the embeddings can be returned
        )

    async def aget_relevant_documents(self):
        return []
