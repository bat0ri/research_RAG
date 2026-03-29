from chromadb.api.models import Collection
from sentence_transformers import SentenceTransformer

class ChromaRetriever:
    collection: Collection
    model: SentenceTransformer = SentenceTransformer('intfloat/multilingual-e5-small')

    def __init__(self, collection):
        self.collection = collection

    def top_n(self, query: str, n:int=3):
        embedding = self.model.encode(["query: " + query]).tolist()
        results = self.collection.query(
            query_embeddings=embedding,
            n_results=n
        )
        
        return results